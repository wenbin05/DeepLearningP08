"""Aggregate saved completed experiments; require all independent seeds by default."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from src.train import write_json


def group_key(result):
    m = result['manifest']
    c = m['config']
    data = m['evaluation'] if result['phase'] == 'test' else m['data']['validation']
    return (c['placement'], c['lora_rank'], c['projections'], c['prefix_tokens'],
            result['phase'], data['dataset'])


def protocol(result):
    m = result['manifest']
    args = dict(m['training_arguments'])
    for field in ('seed', 'data_seed', 'output_dir', 'logging_dir', 'run_name'):
        args.pop(field, None)
    return {'config': m['config'], 'data': m['data'], 'generation': m['generation'],
            'training_arguments': args, 'parameters': m['parameters'],
            'evaluation': m.get('evaluation'), 'metric_protocol': m['metric_protocol'],
            'reference_protocol': m['reference_protocol'], 'metric_revision': m.get('metric_revision'),
            'model_config': m.get('model_config'),
            'environment': {k: m['environment'].get(k) for k in
                ('packages', 'device', 'cuda', 'world_size', 'git_commit', 'git_dirty')},
            'evaluation_environment': m.get('evaluation_environment') and {
                k: m['evaluation_environment'].get(k) for k in
                ('packages', 'device', 'cuda', 'world_size', 'git_commit', 'git_dirty')}}


def aggregate(paths, expected_seeds=(42, 123, 999)):
    groups = {}
    for path in paths:
        r = json.loads(Path(path).read_text())
        if r.get('schema_version') != 1 or r.get('status') != 'completed':
            raise ValueError(f'{path}: not a completed experiment')
        groups.setdefault(group_key(r), []).append((str(path), r))
    if not groups:
        raise ValueError('No saved experiment results found')
    summary = []
    for key, runs in sorted(groups.items()):
        seeds = [r['manifest']['seed'] for _, r in runs]
        if len(set(seeds)) != len(seeds) or set(seeds) != set(expected_seeds):
            raise ValueError(f'{key}: expected one run per seed {expected_seeds}, got {seeds}')
        reference = protocol(runs[0][1])
        if any(protocol(r) != reference for _, r in runs):
            raise ValueError(f'{key}: incompatible protocols; do not pool these runs')
        prefix = key[4]
        metrics = {}
        for metric in ('rouge1', 'rouge2', 'rougeL', 'loss'):
            values = [r['metrics'][f'{prefix}_{metric}'] for _, r in runs]
            metrics[metric] = {'mean': mean(values), 'std': stdev(values) if len(values) > 1 else None}
        if prefix == 'validation':
            values = [r['training_seconds'] for _, r in runs]
            metrics['training_seconds'] = {'mean': mean(values), 'std': stdev(values) if len(values) > 1 else None}
        summary.append({'placement': key[0], 'rank': key[1], 'projections': key[2],
                        'prefix_tokens': key[3], 'phase': key[4], 'dataset': key[5],
                        'seeds': sorted(seeds), 'n': len(seeds), 'metrics': metrics,
                        'parameters': runs[0][1]['manifest']['parameters'],
                        'source_files': [p for p, _ in runs], 'protocol': reference})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results_dir', type=Path, default=Path('results'))
    p.add_argument('--phase', choices=['validation', 'test'], default='test')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999])
    p.add_argument('--output', type=Path, default=Path('results/summary.json'))
    a = p.parse_args()
    write_json(a.output, aggregate(sorted(a.results_dir.rglob(f'{a.phase}_results.json')), a.seeds))
