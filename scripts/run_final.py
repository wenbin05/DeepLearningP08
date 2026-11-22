"""Plan or execute the final study in resumable stages; never resume partial training."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.config import ExperimentConfig
from src.train import environment, write_json

STAGES = {
    'placements': [(m, 8, 'q_proj,v_proj') for m in
                   ('zero_shot', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn')],
    'baselines': [(m, 8, 'q_proj,v_proj') for m in ('all_attn', 'prefix', 'full_ft')],
    'ablations': [('all_attn', r, 'q_proj,v_proj') for r in (4, 16)] +
                 [('all_attn', 8, p) for p in ('q_proj', 'k_proj', 'v_proj',
                                              'q_proj,k_proj,v_proj,out_proj')],
}


def matrix(stage='all'):
    return [entry for name, entries in STAGES.items()
            if stage == 'all' or stage == name for entry in entries]


def read_protocol(path):
    p = json.loads(Path(path).read_text())
    required = {'study', 'model_revision', 'cnn_revision', 'xsum_revision', 'seeds',
                'train_samples', 'validation_samples', 'test_samples', 'epochs',
                'warmup_steps', 'fp16', 'data_seed', 'train_batch_size',
                'gradient_accumulation_steps', 'eval_batch_size'}
    if set(p) != required:
        raise ValueError(f'Protocol must contain exactly {sorted(required)}')
    if not p['seeds'] or len(set(p['seeds'])) != len(p['seeds']):
        raise ValueError('Use distinct training seeds')
    for k in ('model_revision', 'cnn_revision', 'xsum_revision'):
        if len(p[k]) != 40 or any(c not in '0123456789abcdef' for c in p[k]):
            raise ValueError(f'{k} must be an immutable Hub SHA')
    for k in ('train_samples', 'validation_samples', 'test_samples'):
        if p[k] is not None and (not isinstance(p[k], int) or p[k] <= 0):
            raise ValueError(f'{k} must be positive or null for the full split')
    if p['epochs'] <= 0 or p['warmup_steps'] < 0 or not isinstance(p['fp16'], bool):
        raise ValueError('Invalid training schedule or precision')
    if any(not isinstance(p[k], int) or p[k] <= 0 for k in
           ('train_batch_size', 'gradient_accumulation_steps', 'eval_batch_size')):
        raise ValueError('Batch sizes and gradient accumulation must be positive integers')
    return p


def run_name(mode, rank, projections, seed):
    return f'{mode}_r{rank}_{projections}_seed{seed}'


def expected_config(p, mode, rank, projections):
    return dataclasses.asdict(ExperimentConfig(placement=mode, lora_rank=rank,
        projections=projections, model_revision=p['model_revision'],
        dataset_revision=p['cnn_revision'], data_seed=p['data_seed'],
        num_train_epochs=p['epochs'], warmup_steps=p['warmup_steps'], fp16=p['fp16'],
        per_device_train_batch_size=p['train_batch_size'],
        gradient_accumulation_steps=p['gradient_accumulation_steps'],
        per_device_eval_batch_size=p['eval_batch_size']))


def check_training(result, p, mode, rank, projections, seed):
    m = result['manifest']
    if result.get('status') != 'completed' or result.get('phase') != 'validation':
        raise ValueError('Not a completed validation record')
    if m['seed'] != seed or m['config'] != expected_config(p, mode, rank, projections):
        raise ValueError('Saved configuration differs from final protocol')
    if m['training_arguments']['max_steps'] != -1:
        raise ValueError('Final protocol requires the full configured epochs, not a step cap')
    c, args = m['config'], m['training_arguments']
    for name in ('num_train_epochs', 'warmup_steps', 'per_device_train_batch_size',
                 'per_device_eval_batch_size', 'gradient_accumulation_steps', 'fp16',
                 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon'):
        if args.get(name) != c[name]:
            raise ValueError(f'Trainer {name} differs from the declared configuration')
    lr = c['lr_full_ft'] if mode == 'full_ft' else c['learning_rate']
    if args.get('learning_rate') != lr:
        raise ValueError('Trainer learning rate differs from the declared configuration')
    for split, count in (('train', p['train_samples']), ('validation', p['validation_samples'])):
        meta = m['data'][split]
        if split == 'train' and mode == 'zero_shot':
            if meta is not None:
                raise ValueError('Zero-shot unexpectedly has training data')
            continue
        if (meta['split'] != split or meta['revision'] != p['cnn_revision'] or
                meta['samples'] != (meta['official_size'] if count is None else count) or
                meta['data_seed'] != p['data_seed']):
            raise ValueError(f'Saved {split} data differs from final protocol')


def check_evaluation(result, training, p, dataset):
    if result.get('status') != 'completed' or result.get('phase') != 'test':
        raise ValueError('Not a completed test record')
    m = result['manifest']
    for key, value in training['manifest'].items():
        if m.get(key) != value:
            raise ValueError(f'Test record differs from its training manifest: {key}')
    meta = m['evaluation']
    revision = p['cnn_revision'] if dataset == 'cnn_dailymail' else p['xsum_revision']
    if (meta['dataset'] != dataset or meta['split'] != 'test' or meta['revision'] != revision
            or meta['data_seed'] != p['data_seed'] or
            meta['samples'] != (meta['official_size'] if p['test_samples'] is None else p['test_samples'])):
        raise ValueError('Test split, revision or sample count differs from final protocol')


def require_artifacts(directory):
    for name in ('run_config.json', 'trainer_state.json', 'validation_predictions.jsonl', 'model/tokenizer_config.json'):
        if not (directory / name).is_file():
            raise ValueError(f'Missing artifact: {directory / name}')
    if not list((directory / 'model').glob('*.safetensors')):
        raise ValueError(f'Missing saved model/adapter weights: {directory}')


def code_digest():
    root = Path(__file__).resolve().parents[1]
    files = [root / 'requirements.txt', *sorted((root / 'src').glob('*.py')),
             *sorted((root / 'scripts').glob('*.py'))]
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes())
    return h.hexdigest()


def execute(p, root, entries, seeds, phase):
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive launcher lock. An interrupted process leaves the lock for explicit inspection.
    lock = root / '.launcher.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        current = environment()
        identity = {'protocol': p, 'source_sha256': code_digest(),
                    'cpu_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                    'environment': {k: current[k] for k in
                        ('python', 'packages', 'device', 'cuda', 'world_size', 'git_commit', 'git_dirty')}}
        stamp = root / 'study.json'
        if stamp.exists() and json.loads(stamp.read_text()) != identity:
            raise ValueError('Study code, environment or protocol changed; use a separate output directory')
        if not stamp.exists():
            if list(root.glob('*_seed*')):
                raise ValueError('Cannot adopt existing run folders without their study manifest')
            write_json(stamp, identity)

        def invoke(arguments):
            subprocess.run([sys.executable, '-m', 'src.train', *map(str, arguments)], check=True,
                           env=dict(os.environ, PYTHONHASHSEED='0'))

        for mode, rank, projections in entries:
            for seed in seeds:
                directory = root / run_name(mode, rank, projections, seed)
                path = directory / 'validation_results.json'
                if not path.exists():
                    if phase == 'evaluate':
                        raise ValueError(f'Train first: {directory}')
                    if directory.exists():
                        raise ValueError(f'Incomplete run preserved: {directory}. Archive outside the study before retrying.')
                    args = ['--placement', mode, '--seed', seed, '--lora_rank', rank,
                            '--projections', projections, '--output_dir', root,
                            '--model_revision', p['model_revision'], '--dataset_revision', p['cnn_revision'],
                            '--data_seed', p['data_seed'], '--num_train_epochs', p['epochs'],
                            '--warmup_steps', p['warmup_steps'],
                            '--train_batch_size', p['train_batch_size'],
                            '--gradient_accumulation_steps', p['gradient_accumulation_steps'],
                            '--eval_batch_size', p['eval_batch_size'],
                            '--max_train_samples', p['train_samples'] or 'all',
                            '--max_eval_samples', p['validation_samples'] or 'all']
                    invoke(args + (['--fp16'] if p['fp16'] else []))
                training = json.loads(path.read_text())
                check_training(training, p, mode, rank, projections, seed)
                require_artifacts(directory)
                if phase == 'train':
                    continue
                for dataset, revision in [('cnn_dailymail', p['cnn_revision']), ('xsum', p['xsum_revision'])]:
                    files = list((directory / 'evaluations').glob(f'{dataset}_*/test_results.json'))
                    if not files:
                        if list((directory / 'evaluations').glob(f'{dataset}_*')):
                            raise ValueError(f'Incomplete evaluation preserved in {directory}; archive it before retrying')
                        invoke(['--eval_only', directory, '--eval_dataset', dataset,
                                '--eval_revision', revision, '--max_eval_samples', p['test_samples'] or 'all'])
                        files = list((directory / 'evaluations').glob(f'{dataset}_*/test_results.json'))
                    if len(files) != 1:
                        raise ValueError(f'Expected exactly one {dataset} evaluation: {directory}')
                    check_evaluation(json.loads(files[0].read_text()), training, p, dataset)
                    if not files[0].with_name('predictions.jsonl').is_file():
                        raise ValueError(f'Missing test predictions: {files[0]}')
                print('Completed and checked:', directory, flush=True)
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path('experiments/final_protocol.json'))
    parser.add_argument('--output_dir', type=Path, default=Path('results/final_v1'))
    parser.add_argument('--stage', choices=['placements', 'baselines', 'ablations', 'all'], default='placements')
    parser.add_argument('--phase', choices=['train', 'evaluate', 'all'], default='all')
    parser.add_argument('--seed', type=int, help='Run one of the study seeds per session')
    parser.add_argument('--mode', help='Run one mode in the selected stage')
    parser.add_argument('--execute', action='store_true', help='Without this flag, only print the plan')
    a = parser.parse_args()
    p = read_protocol(a.protocol)
    entries = [e for e in matrix(a.stage) if a.mode is None or e[0] == a.mode]
    seeds = p['seeds'] if a.seed is None else [a.seed]
    if not entries or any(s not in p['seeds'] for s in seeds):
        parser.error('Select a mode in this stage and seeds in the study protocol')
    for entry in entries:
        for seed in seeds:
            print(a.phase, a.output_dir / run_name(*entry, seed))
    if a.execute:
        import torch
        if p['fp16'] and not torch.cuda.is_available():
            parser.error('This protocol requires a CUDA GPU; planning is available on CPU')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            parser.error('This protocol supports a single device only')
        execute(p, a.output_dir, entries, seeds, a.phase)
    else:
        print('Plan only. Add --execute to start; keep one launcher active per study.')


if __name__ == '__main__':
    main()
