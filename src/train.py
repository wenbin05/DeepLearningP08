"""Train on CNN/DailyMail validation; evaluate saved runs on held-out test splits."""
import argparse
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import (DataCollatorForSeq2Seq, GenerationConfig, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, set_seed)
from src.config import ExperimentConfig
from src.data import encode, load_split, resolve_revision
from src.evaluate import decode_outputs, rouge_metric, trainer_metrics
from src.model import MODES, load_tokenizer_and_model, parameter_counts


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sample_limit(value):
    if value == "all":
        return None
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive count or all")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--placement', choices=MODES, default='cross_attn')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_seed', type=int, default=42)
    p.add_argument('--max_train_samples', type=sample_limit, default=10000)
    p.add_argument('--max_eval_samples', type=sample_limit, default=None)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--lora_rank', type=int, default=8)
    p.add_argument('--projections', default='q_proj,v_proj')
    p.add_argument('--prefix_tokens', type=int, default=20)
    p.add_argument('--num_train_epochs', type=float, default=3)
    p.add_argument('--max_steps', type=int, default=-1, help='Positive value caps updates for compute-controlled pilots')
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--model_revision', default='main')
    p.add_argument('--dataset_revision', default='main')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--eval_only', type=Path, help='Saved run directory; no training occurs')
    p.add_argument('--eval_dataset', choices=['cnn_dailymail', 'xsum'], default='cnn_dailymail')
    p.add_argument('--eval_revision', default='main', help='Hub dataset revision for held-out evaluation')
    return p.parse_args(argv)


def generation_config(model, config):
    generation = GenerationConfig.from_model_config(model.config)
    generation.update(max_length=config.max_target_length, num_beams=config.num_beams,
                      length_penalty=config.length_penalty, min_length=config.min_length,
                      no_repeat_ngram_size=config.no_repeat_ngram_size, do_sample=False)
    return generation


def training_arguments(config, output, seed, max_steps=-1):
    return Seq2SeqTrainingArguments(
        output_dir=str(output), eval_strategy='epoch', save_strategy='epoch',
        learning_rate=config.lr_full_ft if config.placement == 'full_ft' else config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs, max_steps=max_steps,
        warmup_steps=config.warmup_steps, weight_decay=config.weight_decay,
        adam_beta1=config.adam_beta1, adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon, optim='adamw_torch', lr_scheduler_type='linear',
        max_grad_norm=1.0, fp16=config.fp16, predict_with_generate=True,
        generation_max_length=config.max_target_length, generation_num_beams=config.num_beams,
        save_total_limit=1, seed=seed, data_seed=seed, full_determinism=True,
        report_to='none', logging_steps=50, dataloader_num_workers=0)


def make_trainer(model, tokenizer, config, output, seed, val, train=None, max_steps=-1, references=None):
    model.generation_config = generation_config(model, config)
    # PEFT delegates generation to its base; apply the same settings there as well.
    if hasattr(model, "get_base_model"):
        model.get_base_model().generation_config = model.generation_config
    return Seq2SeqTrainer(
        model=model, args=training_arguments(config, output, seed, max_steps),
        train_dataset=train, eval_dataset=val, tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=trainer_metrics(tokenizer, references))


def environment():
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'git_commit': commit.stdout.strip() or None, 'git_dirty': bool(dirty.stdout.strip()),
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
            'world_size': int(os.environ.get('WORLD_SIZE', '1'))}


def save_predictions(path, prediction, tokenizer, raw, target):
    predictions, references = decode_outputs(tokenizer, prediction.predictions, prediction.label_ids)
    with Path(path).open('w') as out:
        for index, (row, pred, ref) in enumerate(zip(raw, predictions, references)):
            out.write(json.dumps({'index': index, 'id': row.get('id'),
                                  'source': row.get('article', row.get('document')),
                                  'prediction': pred, 'reference_truncated': ref,
                                  'reference_original': row[target]}, ensure_ascii=False) + '\n')


def main(argv=None):
    args = parse_args(argv)
    if args.fp16 and not torch.cuda.is_available():
        raise ValueError('--fp16 requires CUDA')
    if args.eval_only:
        evaluate_checkpoint(args)
        return
    if args.eval_dataset != 'cnn_dailymail':
        raise ValueError('Training is only on CNN/DailyMail; use --eval_only for XSum')
    if args.lora_rank <= 0 or args.prefix_tokens <= 0:
        raise ValueError('Rank and prefix length must be positive')
    config = ExperimentConfig(**{f: getattr(args, f) for f in (
        'placement', 'lora_rank', 'projections', 'prefix_tokens', 'num_train_epochs',
        'warmup_steps', 'data_seed', 'model_revision', 'dataset_revision', 'fp16')})
    config.model_revision = resolve_revision(config.model_name_or_path, config.model_revision)
    config.dataset_revision = resolve_revision('abisee/cnn_dailymail', config.dataset_revision, 'dataset')
    run = Path(args.output_dir) / f'{config.placement}_r{config.lora_rank}_{config.projections}_seed{args.seed}'
    run.mkdir(parents=True, exist_ok=False)  # Never silently overwrite a previous experiment.
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    # Preflight the official metric before spending time on training.
    rouge_metric()
    tokenizer, model = load_tokenizer_and_model(config)
    raw_val, val_meta = load_split('cnn_dailymail', 'validation', args.max_eval_samples,
                                   config.data_seed, config.dataset_revision)
    val = encode(raw_val, tokenizer, config, 'cnn_dailymail')
    train, train_meta = None, None
    if config.placement != 'zero_shot':
        raw_train, train_meta = load_split('cnn_dailymail', 'train', args.max_train_samples,
                                         config.data_seed, config.dataset_revision)
        train = encode(raw_train, tokenizer, config, 'cnn_dailymail')
    trainer = make_trainer(model, tokenizer, config, run, args.seed, val, train, args.max_steps, list(raw_val["highlights"]))
    manifest = {'schema_version': 1, 'config': dataclasses.asdict(config), 'seed': args.seed,
                'parameters': parameter_counts(model), 'targets': getattr(model, 'p08_target_modules', []),
                'data': {'train': train_meta, 'validation': val_meta},
                'generation': model.generation_config.to_dict(), 'training_arguments': trainer.args.to_dict(),
                'environment': environment(), 'checkpoint_selection': 'final epoch/update',
                'metric_protocol': 'HF Evaluate rouge, stemmed macro F1 x100, rougeL (not rougeLsum)',
                'reference_protocol': 'untruncated original dataset references for ROUGE',
                'metric_revision': '1bc20e7e9df922e5dfb597c31a4ea8467a89caff',
                'model_config': model.config.to_dict()}
    write_json(run / 'run_config.json', manifest)
    started = time.perf_counter()
    train_metrics = trainer.train().metrics if train is not None else {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    duration = time.perf_counter() - started if train is not None else 0.0
    trainer.save_model(str(run / 'model'))
    tokenizer.save_pretrained(run / 'model')
    trainer.save_state()
    prediction = trainer.predict(val, metric_key_prefix='validation')
    metrics = dict(prediction.metrics)
    save_predictions(run / 'validation_predictions.jsonl', prediction, tokenizer, raw_val, 'highlights')
    write_json(run / 'validation_results.json', {'schema_version': 1, 'status': 'completed',
        'phase': 'validation', 'manifest': manifest, 'metrics': metrics,
        'training_metrics': train_metrics, 'training_seconds': duration})


def evaluate_checkpoint(args):
    """Reuse exact base revision and adapter/full checkpoint, with no optimization."""
    from peft import PeftModel
    from transformers import BartForConditionalGeneration, BartTokenizerFast
    run = args.eval_only
    original = json.loads((run / 'run_config.json').read_text())
    config = ExperimentConfig(**original['config'])
    set_seed(original['seed'])
    torch.use_deterministic_algorithms(True)
    rouge_metric()
    checkpoint = run / 'model'
    tokenizer = BartTokenizerFast.from_pretrained(checkpoint)
    if (checkpoint / 'adapter_config.json').exists():
        base = BartForConditionalGeneration.from_pretrained(config.model_name_or_path,
            revision=config.model_revision, attn_implementation='eager')
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    else:
        model = BartForConditionalGeneration.from_pretrained(checkpoint, attn_implementation='eager')
    model.requires_grad_(False)
    raw, meta = load_split(args.eval_dataset, 'test', args.max_eval_samples, config.data_seed, args.eval_revision)
    output = run / 'evaluations' / f'{args.eval_dataset}_{meta["ids_sha256"][:12]}'
    output.mkdir(parents=True, exist_ok=False)
    val = encode(raw, tokenizer, config, args.eval_dataset)
    target = 'summary' if args.eval_dataset == 'xsum' else 'highlights'
    trainer = make_trainer(model, tokenizer, config, output, original['seed'], val, references=list(raw[target]))
    prediction = trainer.predict(val, metric_key_prefix='test')
    metrics = dict(prediction.metrics)
    save_predictions(output / 'predictions.jsonl', prediction, tokenizer, raw, target)
    manifest = dict(original, evaluation=meta, evaluation_environment=environment())
    write_json(output / 'test_results.json', {'schema_version': 1, 'status': 'completed',
               'phase': 'test', 'manifest': manifest, 'metrics': metrics})


if __name__ == '__main__':
    main()
