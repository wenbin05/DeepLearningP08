"""Synthetic evidence fixtures verify that finalization rejects invalid studies."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts.check_final_results import check, rescore
from scripts.human_evaluation import FIELDS, analyze, prepare
from scripts.run_final import (check_training, expected_config, matrix, read_protocol, run_name)
from src.aggregate_results import aggregate
from src.comparison import validate_comparison
from src.train import write_json


@pytest.fixture
def study(tmp_path):
    p = read_protocol('experiments/final_protocol.json')
    p.update(seeds=[42], train_samples=5, validation_samples=5, test_samples=5)
    rows = [{'index': i, 'id': str(i), 'source': 'the cat sat on the mat',
             'prediction': 'the cat sat', 'reference_original': 'the cat sat'} for i in range(5)]
    digest = hashlib.sha256(''.join(f'{i}\n' for i in range(5)).encode()).hexdigest()
    def meta(dataset, split):
        return {'dataset': dataset, 'split': split, 'samples': 5, 'official_size': 50,
                'revision': p['xsum_revision'] if dataset == 'xsum' else p['cnn_revision'],
                'data_seed': 42, 'ids_sha256': digest}
    predictions = []
    for mode, rank, projections in matrix('placements'):
        directory = tmp_path / run_name(mode, rank, projections, 42)
        manifest = {'config': expected_config(p, mode, rank, projections), 'seed': 42,
            'parameters': {'trainable': 0 if mode == 'zero_shot' else 147456},
            'data': {'train': None if mode == 'zero_shot' else meta('cnn_dailymail', 'train'),
                     'validation': meta('cnn_dailymail', 'validation')},
            'generation': {'num_beams': 4}, 'training_arguments': {'max_steps': -1, 'learning_rate': 3e-4},
            'metric_protocol': 'HF Evaluate rouge', 'reference_protocol': 'full',
            'environment': {'packages': {}}, 'checkpoint_selection': 'final epoch/update'}
        for name in ('num_train_epochs', 'warmup_steps', 'per_device_train_batch_size',
                     'per_device_eval_batch_size', 'gradient_accumulation_steps', 'fp16',
                     'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon'):
            manifest['training_arguments'][name] = manifest['config'][name]
        result = {'schema_version': 1, 'status': 'completed', 'phase': 'validation',
                  'manifest': manifest, 'training_seconds': 0 if mode == 'zero_shot' else 1,
                  'metrics': {f'validation_{k}': v for k, v in
                              [('rouge1', 100.), ('rouge2', 100.), ('rougeL', 100.), ('loss', 1.)]}}
        write_json(directory / 'validation_results.json', result)
        write_json(directory / 'run_config.json', manifest)
        write_json(directory / 'trainer_state.json', {'global_step': 0 if mode == 'zero_shot' else 1})
        write_json(directory / 'model' / 'tokenizer_config.json', {})
        (directory / 'model' / 'adapter_model.safetensors').write_bytes(b'synthetic-existence-fixture')
        (directory / 'validation_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        for dataset in ('cnn_dailymail', 'xsum'):
            test = copy.deepcopy(result)
            test['phase'] = 'test'
            test['manifest']['evaluation'] = meta(dataset, 'test')
            test['metrics'] = {k.replace('validation_', 'test_'): v for k, v in result['metrics'].items()}
            path = directory / 'evaluations' / f'{dataset}_fixture' / 'test_results.json'
            write_json(path, test)
            prediction = path.with_name('predictions.jsonl')
            prediction.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            if dataset == 'cnn_dailymail':
                predictions.append(prediction)
    return tmp_path, p, predictions


def test_complete_scoped_matrix_and_rescore(study):
    root, protocol, _ = study
    report, summaries = check(root, protocol, 'placements', recompute=True)
    assert report['experiment_matrix_complete'] and report['rouge_recomputed']
    assert not report['human_study_checked']
    assert len(summaries['test']) == 8
    report, _ = check(root, protocol, 'all')
    assert not report['experiment_matrix_complete'] and len(report['problems']) == 9


def test_missing_artifacts_and_wrong_schedule_fail(study):
    root, p, predictions = study
    directory = predictions[0].parents[2]
    (directory / 'model' / 'adapter_model.safetensors').unlink()
    report, _ = check(root, p, 'placements')
    assert 'weights' in report['problems'][0]['problem']
    path = predictions[1].parents[2] / 'validation_results.json'
    result = json.loads(path.read_text())
    result['manifest']['training_arguments']['max_steps'] = 32
    mode = result['manifest']['config']['placement']
    with pytest.raises(ValueError, match='step cap'):
        check_training(result, p, mode, 8, 'q_proj,v_proj', 42)


@pytest.mark.parametrize('change', ['epochs', 'steps', 'precision', 'revision', 'count', 'seed'])
def test_cross_placement_guard(study, change):
    root, _, _ = study
    rows = aggregate(list(root.glob('*/validation_results.json')), expected_seeds=(42,))
    changed = next(r for r in rows if r['placement'] == 'cross_attn')
    if change == 'epochs':
        changed['protocol']['config']['num_train_epochs'] = 99
    elif change == 'steps':
        changed['protocol']['training_arguments']['max_steps'] = 9999
    elif change == 'precision':
        changed['protocol']['config']['fp16'] = False
    elif change == 'revision':
        changed['protocol']['config']['model_revision'] = 'other'
    elif change == 'count':
        changed['parameters']['trainable'] = 999
    else:
        changed['seeds'] = [999]
    with pytest.raises(ValueError):
        validate_comparison(rows)


def test_rescore_rejects_order_and_score_changes(study):
    _, _, paths = study
    path = paths[0]
    result = json.loads(path.with_name('test_results.json').read_text())
    result['metrics']['test_rouge1'] = float('nan')
    with pytest.raises(ValueError, match='differs'):
        rescore(path, result)
    rows = path.read_text().splitlines()
    path.write_text('\n'.join(reversed(rows)) + '\n')
    with pytest.raises(ValueError, match='row'):
        rescore(path, result)


def test_human_packets_and_complete_anonymous_ratings(study):
    root, _, paths = study
    output = root / 'human'
    key = prepare(paths, output)
    assert len(key) == 20
    files = sorted((output / 'participants').glob('*/ratings.csv'))
    assert len(files) == 10
    with pytest.raises(ValueError, match='Missing/invalid'):
        analyze(output / 'coordinator/key.json', files)
    for path in files:
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 20
        for row in rows:
            row['judgment'] = 'consistent'
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        packet = path.with_name('packet.html').read_text()
        assert all(mode not in packet for mode, _, _ in matrix('placements'))
    report = analyze(output / 'coordinator/key.json', files)
    assert report['ratings'] == 200 and report['pairwise_agreement'] == 1
    assert report['fleiss_kappa_three_categories'] is None
    with pytest.raises(ValueError, match='duplicate'):
        analyze(output / 'coordinator/key.json', files + files[:1])
    with pytest.raises(ValueError, match='received 180'):
        analyze(output / 'coordinator/key.json', files[:-1])


def test_protocol_requires_immutable_revisions(tmp_path):
    p = read_protocol('experiments/final_protocol.json')
    p['model_revision'] = 'main'
    file = tmp_path / 'protocol.json'
    write_json(file, p)
    with pytest.raises(ValueError, match='immutable'):
        read_protocol(file)


def test_launcher_skips_verified_runs_and_preserves_incomplete_ones(study, monkeypatch):
    import scripts.run_final as launcher
    root, p, paths = study
    env = {k: 'test' for k in ('python', 'device', 'cuda', 'world_size', 'git_commit', 'git_dirty')}
    env['packages'] = {}
    monkeypatch.setattr(launcher, 'environment', lambda: env)
    monkeypatch.setattr(launcher, 'code_digest', lambda: 'fixture-digest')
    write_json(root / 'study.json', {'protocol': p, 'source_sha256': 'fixture-digest',
        'environment': env, 'cpu_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}})
    def forbid_run(*args, **kwargs):
        raise AssertionError('A completed run should not be executed again')
    monkeypatch.setattr(launcher.subprocess, 'run', forbid_run)
    launcher.execute(p, root, matrix('placements'), [42], 'all')
    assert not (root / '.launcher.lock').exists()
    validation = paths[0].parents[2] / 'validation_results.json'
    validation.unlink()
    with pytest.raises(ValueError, match='Incomplete run preserved'):
        launcher.execute(p, root, matrix('placements'), [42], 'all')
    assert validation.parent.exists() and not (root / '.launcher.lock').exists()
    changed = dict(p, warmup_steps=99)
    with pytest.raises(ValueError, match='protocol changed'):
        launcher.execute(changed, root, matrix('placements'), [42], 'all')
    (root / '.launcher.lock').touch()
    with pytest.raises(FileExistsError):
        launcher.execute(p, root, matrix('placements'), [42], 'all')
    assert (root / '.launcher.lock').exists()
