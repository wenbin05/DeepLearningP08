"""Check the expected experiment matrix and optionally rescore retained predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.run_final import (check_evaluation, check_training, matrix, read_protocol,
                               require_artifacts, run_name)
from src.aggregate_results import aggregate
from src.comparison import validate_comparison
from src.evaluate import compute_rouge_metrics
from src.train import write_json


def rescore(path, result):
    phase = result['phase']
    meta = result['manifest']['evaluation'] if phase == 'test' else result['manifest']['data']['validation']
    predictions, references = [], []
    digest = hashlib.sha256()
    with Path(path).open() as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row['index'] != index or not all(isinstance(row[k], str) for k in
                    ('source', 'prediction', 'reference_original')):
                raise ValueError(f'Invalid prediction row {index}: {path}')
            identity = row['id'] if row.get('id') is not None else row['source']
            digest.update((str(identity) + '\n').encode())
            predictions.append(row['prediction'])
            references.append(row['reference_original'])
    if len(predictions) != meta['samples'] or digest.hexdigest() != meta['ids_sha256']:
        raise ValueError(f'Prediction count/order differs from evaluation manifest: {path}')
    scores = compute_rouge_metrics(predictions, references)
    for name, value in scores.items():
        recorded = result['metrics'][f'{phase}_{name}']
        if not math.isfinite(recorded) or abs(value - recorded) > 1e-6:
            raise ValueError(f'Recomputed {name} differs from saved result: {path}')
    return scores


def check(root, p, stage='all', recompute=False):
    root = Path(root)
    problems, paths = [], {'validation': [], 'test': []}
    expected = [(e, seed) for e in matrix(stage) for seed in p['seeds']]
    for (mode, rank, projections), seed in expected:
        directory = root / run_name(mode, rank, projections, seed)
        try:
            path = directory / 'validation_results.json'
            training = json.loads(path.read_text())
            check_training(training, p, mode, rank, projections, seed)
            require_artifacts(directory)
            if json.loads((directory / 'run_config.json').read_text()) != training['manifest']:
                raise ValueError('Standalone run manifest differs from result')
            if recompute:
                rescore(directory / 'validation_predictions.jsonl', training)
            paths['validation'].append(path)
            for dataset in ('cnn_dailymail', 'xsum'):
                candidates = list((directory / 'evaluations').glob(f'{dataset}_*/test_results.json'))
                if len(candidates) != 1:
                    raise ValueError(f'Expected one completed {dataset} test evaluation; found {len(candidates)}')
                result = json.loads(candidates[0].read_text())
                check_evaluation(result, training, p, dataset)
                predictions = candidates[0].with_name('predictions.jsonl')
                if not predictions.is_file():
                    raise ValueError(f'Missing predictions: {predictions}')
                if recompute:
                    rescore(predictions, result)
                paths['test'].append(candidates[0])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append({'run': str(directory), 'problem': str(exc)})
    summaries = {}
    if not problems:
        for phase, files in paths.items():
            try:
                summaries[phase] = aggregate(files, expected_seeds=p['seeds'])
                for dataset in {r['dataset'] for r in summaries[phase]}:
                    validate_comparison([r for r in summaries[phase] if r['dataset'] == dataset])
            except (ValueError, KeyError, TypeError) as exc:
                problems.append({'phase': phase, 'problem': str(exc)})
    return {'study': p['study'], 'stage': stage, 'expected_runs': len(expected),
            'experiment_matrix_complete': not problems, 'rouge_recomputed': recompute and not problems,
            'human_study_checked': False, 'problems': problems}, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', type=Path, default=Path('results/final_v1'))
    parser.add_argument('--protocol', type=Path, default=Path('experiments/final_protocol.json'))
    parser.add_argument('--stage', choices=['placements', 'baselines', 'ablations', 'all'], default='all')
    parser.add_argument('--rescore', action='store_true')
    parser.add_argument('--output_dir', type=Path, default=Path('results/final_checks'))
    a = parser.parse_args()
    report, summaries = check(a.results_dir, read_protocol(a.protocol), a.stage, a.rescore)
    write_json(a.output_dir / 'readiness.json', report)
    # Use a fresh output directory per check, or remove outdated summaries on failure.
    for phase in ('validation', 'test'):
        target = a.output_dir / f'{phase}_summary.json'
        if report['experiment_matrix_complete']:
            write_json(target, summaries[phase])
        elif target.exists():
            target.unlink()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['experiment_matrix_complete'] else 1)


if __name__ == '__main__':
    main()
