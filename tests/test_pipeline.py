"""Real PEFT architecture checks and offline Trainer/save/reload smoke coverage."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset
from peft import PeftModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast, set_seed

from src.config import ExperimentConfig, get_target_modules
from src.count_params import bart_base_config, verify_live_model
from src.data import encode, load_split
from src.evaluate import compute_rouge_metrics, decode_outputs
from src.model import PLACEMENTS, configure_model, parameter_counts
from src.train import generation_config, make_trainer, write_json


def test_live_bart_base_budgets():
    counts = verify_live_model()
    assert counts['full_ft']['total'] == 139420416
    assert counts['zero_shot']['trainable'] == 0
    assert counts['all_attn']['trainable'] == 442368
    assert {counts[p]['trainable'] for p in PLACEMENTS} == {147456}


@pytest.mark.parametrize('placement', PLACEMENTS)
@pytest.mark.parametrize('rank', [4, 8, 16])
def test_actual_lora_targets_and_frozen_base(placement, rank):
    # Real BART-base module tree and PEFT injection; meta tensors avoid allocating 140M weights.
    with torch.device('meta'):
        model = configure_model(BartForConditionalGeneration(bart_base_config()),
                                ExperimentConfig(placement=placement, lora_rank=rank))
    targets = get_target_modules(placement)
    assert set(model.p08_target_modules) == set(targets)
    assert parameter_counts(model)['trainable'] == 6 * 2 * 2 * 768 * rank
    assert all('lora_' in name for name, p in model.named_parameters() if p.requires_grad)
    assert len([name for name, p in model.named_parameters() if p.requires_grad]) == 24


@pytest.mark.parametrize('projections', ['q_proj', 'k_proj', 'v_proj', 'q_proj,k_proj,v_proj,out_proj'])
def test_projection_ablation_budget(projections):
    with torch.device('meta'):
        model = configure_model(BartForConditionalGeneration(bart_base_config()),
            ExperimentConfig(placement='all_attn', projections=projections))
    assert parameter_counts(model)['trainable'] == 18 * len(projections.split(',')) * 2 * 768 * 8


def test_invalid_targets():
    for bad in ['q_proj,q_proj', 'invalid', '']:
        with pytest.raises(ValueError):
            get_target_modules('cross_attn', bad)
    with pytest.raises(ValueError):
        get_target_modules('unknown')


def test_official_rouge_and_repetition_regression():
    # Real official Evaluate metric; no stand-in scoring implementation.
    assert compute_rouge_metrics(['the cat sat'], ['the cat sat']) == {
        'rouge1': 100.0, 'rouge2': 100.0, 'rougeL': 100.0}
    scores = compute_rouge_metrics(['cat cat cat cat'], ['cat'])
    assert scores['rouge1'] == pytest.approx(40)
    assert scores['rougeL'] == pytest.approx(40)
    assert scores['rouge2'] == 0
    assert compute_rouge_metrics([''], ['cat'])['rougeL'] == 0
    with pytest.raises(ValueError):
        compute_rouge_metrics(['a'], ['a', 'b'])
    with pytest.raises(ValueError):
        compute_rouge_metrics([], [])


def test_metric_failure_propagates(monkeypatch):
    import src.evaluate as evaluation
    def fail():
        raise RuntimeError('metric unavailable')
    monkeypatch.setattr(evaluation, 'rouge_metric', fail)
    with pytest.raises(RuntimeError, match='metric unavailable'):
        evaluation.compute_rouge_metrics(['a'], ['a'])


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3,
                                  'cat': 4, 'sat': 5, 'dog': 6}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='<s>',
        eos_token='</s>', unk_token='<unk>', pad_token='<pad>')


def test_preprocessing_and_padding(tokenizer):
    raw = Dataset.from_dict({'article': ['cat sat dog cat', 'dog'],
                             'highlights': ['cat sat dog', 'dog']})
    config = ExperimentConfig(max_source_length=3, max_target_length=2)
    encoded = encode(raw, tokenizer, config, 'cnn_dailymail')
    assert encoded.column_names == ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    assert encoded[0]['input_ids'] == [4, 5, 6]
    assert encoded[0]['labels'] == [4, 5]
    assert encoded[1]['labels'] == [6]
    pred, refs = decode_outputs(tokenizer, np.array([[4, 1]]), np.array([[4, -100]]))
    assert pred == refs == ['cat']
    xsum = Dataset.from_dict({'document': ['cat sat'], 'summary': ['cat']})
    assert encode(xsum, tokenizer, config, 'xsum')[0]['labels'] == [4]


def test_split_selection_is_fixed_and_logged(monkeypatch):
    import src.data as data
    raw = Dataset.from_dict({'id': [str(i) for i in range(20)],
                             'article': ['cat'] * 20, 'highlights': ['cat'] * 20})
    monkeypatch.setattr(data, 'resolve_revision', lambda *a: 'immutable-sha')
    seen = []
    def loader(*args, **kwargs):
        seen.append(kwargs)
        return raw
    monkeypatch.setattr(data, 'load_dataset', loader)
    a, ma = load_split('cnn_dailymail', 'validation', 4, 42)
    b, mb = load_split('cnn_dailymail', 'validation', 4, 42)
    c, mc = load_split('cnn_dailymail', 'validation', 4, 123)
    assert a['id'] == b['id'] and ma == mb
    assert a['id'] != c['id'] and ma['ids_sha256'] != mc['ids_sha256']
    assert seen[0] == {'split': 'validation', 'revision': 'immutable-sha'}
    for limit in (0, -1, 21):
        with pytest.raises(ValueError):
            load_split('cnn_dailymail', 'test', limit, 42)


@pytest.mark.parametrize('mode', ['full_ft', 'prefix', 'zero_shot', *PLACEMENTS])
def test_trainer_checkpoint_smoke(mode, tokenizer, tmp_path):
    torch.set_num_threads(1)
    set_seed(42)
    tiny = BartConfig(vocab_size=7, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, decoder_start_token_id=2)
    config = ExperimentConfig(placement=mode, prefix_tokens=2, max_source_length=8,
        max_target_length=6, min_length=2, num_beams=2, warmup_steps=0,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, num_train_epochs=1)
    if mode in PLACEMENTS:
        tiny = bart_base_config()
        tiny.vocab_size = 7
        tiny.max_position_embeddings = 32
    model = configure_model(BartForConditionalGeneration(tiny), config)
    raw = Dataset.from_dict({'article': ['cat sat', 'dog sat'], 'highlights': ['cat', 'dog']})
    encoded = encode(raw, tokenizer, config, 'cnn_dailymail')
    # Fast tokenizer above emits token_type_ids; Bart does not accept them.
    encoded = encoded.remove_columns('token_type_ids')
    trainer = make_trainer(model, tokenizer, config, tmp_path, 42, encoded,
        None if mode == 'zero_shot' else encoded, max_steps=1,
        references=list(raw['highlights']))
    if mode != 'zero_shot':
        before_parameters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        trainer.train()
        assert any(not torch.equal(before_parameters[n], p) for n, p in model.named_parameters() if p.requires_grad)
        assert trainer.state.global_step == 1
    if hasattr(model, 'get_base_model'):
        assert model.get_base_model().generation_config is model.generation_config
    prediction = trainer.predict(encoded, metric_key_prefix='validation')
    assert all(f'validation_{k}' in prediction.metrics for k in ('rouge1', 'rouge2', 'rougeL', 'loss'))
    assert all(np.isfinite(v) for v in prediction.metrics.values())
    checkpoint = tmp_path / 'model'
    trainer.save_model(str(checkpoint))
    tokenizer.save_pretrained(checkpoint)
    if mode == 'prefix' or mode in PLACEMENTS:
        # Identical random base weights emulate reloading the same pinned pretrained revision.
        set_seed(42)
        reloaded = PeftModel.from_pretrained(BartForConditionalGeneration(tiny), checkpoint)
    else:
        reloaded = BartForConditionalGeneration.from_pretrained(checkpoint)
    model.eval(); reloaded.eval()
    inputs = torch.tensor([[4, 5]])
    with torch.no_grad():
        before = model.generate(input_ids=inputs, generation_config=generation_config(model, config))
        after = reloaded.generate(input_ids=inputs, generation_config=generation_config(reloaded, config))
    assert torch.equal(before, after)
    write_json(tmp_path / 'metrics.json', prediction.metrics)
    assert json.loads((tmp_path / 'metrics.json').read_text())['validation_loss'] >= 0


def test_cli_artifacts_and_cross_domain_no_training(monkeypatch, tokenizer, tmp_path):
    """Exercise production CLI orchestration with local random weights and synthetic rows."""
    import src.train as training
    torch.set_num_threads(1)
    base = tmp_path / 'base'
    tiny = BartConfig(vocab_size=7, d_model=16, encoder_layers=1, decoder_layers=1,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=32, decoder_start_token_id=2)
    BartForConditionalGeneration(tiny).save_pretrained(base)
    tokenizer.save_pretrained(base)
    def config_factory(**kwargs):
        return ExperimentConfig(model_name_or_path=str(base), max_source_length=8,
            max_target_length=6, min_length=2, num_beams=2,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1, **kwargs)
    monkeypatch.setattr(training, 'ExperimentConfig', config_factory)
    # evaluate_checkpoint restores via ExperimentConfig(**saved_config), so restore the
    # real dataclass before invoking it below.
    monkeypatch.setattr(training, 'resolve_revision', lambda *a: 'local-fixture-revision')
    seen = []
    def split(name, part, limit, data_seed, revision):
        seen.append((name, part))
        columns = {'id': ['fixture-1', 'fixture-2']}
        columns.update({'document': ['cat sat', 'dog sat'], 'summary': ['cat', 'dog']}
                       if name == 'xsum' else
                       {'article': ['cat sat', 'dog sat'], 'highlights': ['cat', 'dog']})
        return Dataset.from_dict(columns), {'dataset': name, 'split': part, 'samples': 2,
                                             'ids_sha256': name + part}
    monkeypatch.setattr(training, 'load_split', split)
    output = tmp_path / 'runs'
    training.main(['--placement', 'full_ft', '--output_dir', str(output),
                   '--max_steps', '1', '--warmup_steps', '0'])
    run = next(output.iterdir())
    result = json.loads((run / 'validation_results.json').read_text())
    assert result['status'] == 'completed' and result['training_seconds'] > 0
    assert result['manifest']['parameters']['trainable'] > 0
    assert result['metrics']['validation_loss'] >= 0
    assert (run / 'trainer_state.json').exists()
    assert (run / 'model' / 'model.safetensors').exists()
    assert ('cnn_dailymail', 'train') in seen
    monkeypatch.setattr(training, 'ExperimentConfig', ExperimentConfig)
    # Prove test evaluation never enters a training/optimizer loop.
    def forbid_train(*args, **kwargs):
        raise AssertionError('test evaluation attempted fine-tuning')
    monkeypatch.setattr(training.Seq2SeqTrainer, 'train', forbid_train)
    for name in ['cnn_dailymail', 'xsum']:
        training.main(['--eval_only', str(run), '--eval_dataset', name])
        path = next((run / 'evaluations').glob(f'{name}_*/test_results.json'))
        evaluated = json.loads(path.read_text())
        assert evaluated['phase'] == 'test'
        assert evaluated['manifest']['evaluation']['split'] == 'test'
        assert evaluated['metrics']['test_loss'] >= 0
        predictions = [json.loads(line) for line in path.with_name('predictions.jsonl').read_text().splitlines()]
        assert len(predictions) == 2 and predictions[0]['source'] == 'cat sat'
    assert seen[-2:] == [('cnn_dailymail', 'test'), ('xsum', 'test')]
