"""Validate the supplied pilot export and regenerate its report artifacts.

Run from the repository root with: python -m scripts.prepare_midterm_report
This checks saved records; ROUGE cannot be recomputed without predictions.
"""
import hashlib
import json
import math
from pathlib import Path

from src.aggregate_results import aggregate
from src.config import get_target_modules
from src.train import write_json

ROOT = Path('experiments/midterm_pilot_seed42')
LABELS = {'zero_shot': 'Zero-shot BART-base',
          'encoder_self_attn': 'Encoder self-attention',
          'decoder_self_attn': 'Decoder self-attention', 'cross_attn': 'Cross-attention'}


def prepare():
    bundle = ROOT / 'completed_pilot_outputs.json'
    records = json.loads(bundle.read_text())
    by_mode = {r['result']['manifest']['config']['placement']: r['result'] for r in records}
    if len(records) != 4 or set(by_mode) != set(LABELS):
        raise ValueError('Expected one completed record for zero-shot and each isolated placement')
    common = by_mode['cross_attn']['manifest']
    common_config = {k: v for k, v in common['config'].items() if k != 'placement'}
    common_args = {k: v for k, v in common['training_arguments'].items()
                   if k not in ('output_dir', 'logging_dir', 'run_name')}
    markdown = (ROOT / 'preliminary_table.md').read_text()
    paths, table_rows, checks = [], [], {}
    for mode in LABELS:
        r = by_mode[mode]
        m, c, metrics = r['manifest'], r['manifest']['config'], r['metrics']
        assert r['schema_version'] == 1 and r['status'] == 'completed' and r['phase'] == 'validation'
        assert m['seed'] == 42 and c['data_seed'] == 42
        assert c['lora_rank'] == 8 and c['projections'] == 'q_proj,v_proj' and c['fp16'] is True
        assert c['warmup_steps'] == 4 and c['num_train_epochs'] == 1
        assert m['training_arguments']['max_steps'] == 32
        assert {k: v for k, v in c.items() if k != 'placement'} == common_config
        assert {k: v for k, v in m['training_arguments'].items()
                if k not in ('output_dir', 'logging_dir', 'run_name')} == common_args
        for key in ('generation', 'model_config', 'metric_protocol', 'metric_revision',
                    'reference_protocol', 'environment'):
            assert m[key] == common[key], f'{mode}: different {key}'
        assert m['data']['validation'] == common['data']['validation']
        assert m['data']['validation']['split'] == 'validation'
        assert m['data']['validation']['samples'] == 64
        expected = 0 if mode == 'zero_shot' else 147456
        assert m['parameters']['trainable'] == expected
        assert m['parameters']['total'] == 139420416 + expected
        if mode == 'zero_shot':
            assert m['data']['train'] is None and r['training_seconds'] == 0
            assert r['training_metrics'] == {} and m['targets'] == []
        else:
            assert m['data']['train'] == common['data']['train']
            assert m['data']['train']['split'] == 'train' and m['data']['train']['samples'] == 1024
            assert set(m['targets']) == set(get_target_modules(mode))
            assert r['training_metrics']['epoch'] == 1
            stats = r['training_metrics']
            assert round(stats['train_runtime'] * stats['train_steps_per_second']) == 32
        for key in ('rouge1', 'rouge2', 'rougeL'):
            assert math.isfinite(metrics[f'validation_{key}']) and 0 <= metrics[f'validation_{key}'] <= 100
        assert math.isfinite(metrics['validation_loss']) and metrics['validation_loss'] >= 0
        assert math.isfinite(r['training_seconds']) and r['training_seconds'] >= 0
        values = [f'{metrics[f"validation_{key}"]:.2f}' for key in ('rouge1', 'rouge2', 'rougeL')]
        values += [f'{metrics["validation_loss"]:.4f}', f'{r["training_seconds"]:.1f}']
        markdown_row = '| ' + ' | '.join([mode, f'{expected:,}', *values]) + ' |'
        assert markdown_row in markdown, f'{mode}: uploaded Markdown differs from JSON'
        table_rows.append(' & '.join([LABELS[mode], f'{expected:,}', *values]) + r' \\')
        path = ROOT / 'runs' / mode / 'validation_results.json'
        write_json(path, r)
        paths.append(path)
        checks[mode] = {'status': 'completed', 'seed': 42, 'configured_updates': 0 if mode == 'zero_shot' else 32}
    table = '\n'.join([
        r'\begin{table}[ht]', r'\centering\small',
        r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
        r'Configuration & Trainable & R-1 & R-2 & R-L & Val. loss & Loop s \\',
        r'\midrule', *table_rows, r'\bottomrule', r'\end{tabular}',
        r'\caption{Single-seed pilot on 64 CNN/DailyMail validation examples (seed 42). LoRA runs use 1,024 training examples and 32 updates. Scores are ROUGE F1 $\times100$; no standard deviations are available. Loop time includes epoch validation and checkpoint saving.}',
        r'\label{tab:prelim_results}', r'\end{table}', ''])
    (ROOT / 'results_table.tex').write_text(table)
    write_json(ROOT / 'summary.json', aggregate(paths, expected_seeds=(42,)))
    write_json(ROOT / 'verification.json', {
        'input_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
        'markdown_matches_json': True, 'common_protocol_and_data_hashes': True,
        'records': checks, 'independent_rouge_recomputation': False,
        'limitation': 'Predictions, checkpoints, per-example scores and test/transfer results were not supplied.'})
    print('Validated four supplied records; regenerated table, per-run JSON, and single-seed summary.')


if __name__ == '__main__':
    prepare()
