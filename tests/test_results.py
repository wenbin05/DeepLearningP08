"""Test statistics and provenance guards with explicitly synthetic test fixtures."""
import copy
import json

import pytest

from src.aggregate_results import aggregate
from src.plot_results import generate_pareto_plot
from src.train import write_json


@pytest.fixture
def outputs(tmp_path):
    # Synthetic unit-test inputs only: these are never published as experiment results.
    paths = []
    for seed, value in zip([42, 123, 999], [1., 2., 3.]):
        r = {'schema_version': 1, 'status': 'completed', 'phase': 'test',
             'manifest': {'seed': seed, 'config': {'placement': 'cross_attn', 'lora_rank': 8,
                'projections': 'q_proj,v_proj', 'prefix_tokens': 20},
                'data': {'train': {'ids_sha256': 'train'}, 'validation': {'ids_sha256': 'val'}},
                'evaluation': {'dataset': 'cnn_dailymail', 'ids_sha256': 'test'},
                'generation': {'num_beams': 4}, 'parameters': {'trainable': 147456},
                'training_arguments': {'seed': seed, 'data_seed': seed, 'output_dir': str(seed)},
                'metric_protocol': 'test fixture', 'reference_protocol': 'full',
                'environment': {'packages': {}}},
             'metrics': {f'test_{k}': value for k in ('rouge1', 'rouge2', 'rougeL', 'loss')}}
        path = tmp_path / f'{seed}.json'
        write_json(path, r)
        paths.append(path)
    return paths


def test_genuine_file_loading_statistics_and_plot(outputs, tmp_path):
    result = aggregate(outputs)
    assert result[0]['metrics']['rouge1'] == {'mean': 2., 'std': 1.}
    assert result[0]['seeds'] == [42, 123, 999]
    summary = tmp_path / 'summary.json'
    write_json(summary, result)
    figure = tmp_path / 'plot.png'
    generate_pareto_plot(summary, figure)
    assert figure.read_bytes().startswith(b'\x89PNG')


def test_missing_duplicate_incompatible_and_empty(outputs):
    with pytest.raises(ValueError, match='expected one run'):
        aggregate(outputs[:2])
    with pytest.raises(ValueError, match='expected one run'):
        aggregate(outputs + outputs[:1])
    r = json.loads(outputs[0].read_text())
    r['manifest']['generation']['num_beams'] = 1
    write_json(outputs[0], r)
    with pytest.raises(ValueError, match='incompatible protocols'):
        aggregate(outputs)
    with pytest.raises(ValueError, match='No saved experiment'):
        aggregate([])


def test_plot_refuses_incompatible_evaluations(outputs, tmp_path):
    rows = aggregate(outputs)
    other = copy.deepcopy(rows[0])
    other['protocol']['evaluation']['ids_sha256'] = 'different-test-subset'
    rows.append(other)
    path = tmp_path / 'summary.json'
    write_json(path, rows)
    with pytest.raises(ValueError, match='Incompatible data'):
        generate_pareto_plot(path, tmp_path / 'bad.png')
