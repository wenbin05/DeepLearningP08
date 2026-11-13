"""Recompute descriptive statistics from the unchanged submitted survey CSVs.
Run from any directory: python survey_results/analyze.py
Uses only the Python standard library. Raw responses are never modified.
"""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter, defaultdict

MODES = {'zero_shot', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn'}
LABELS = ('consistent', 'inconsistent', 'uncertain')

def analyze(key_path, rating_paths):
    key = json.loads(Path(key_path).read_text())
    if len(key) != 20 or set(item['mode'] for item in key.values()) != MODES:
        raise ValueError('Expected the 20-summary, four-configuration study key')
    if Counter(item['mode'] for item in key.values()) != Counter({m: 5 for m in MODES}):
        raise ValueError('Expected five summaries per configuration')
    seen, votes = set(), defaultdict(Counter)
    for path in rating_paths:
        with Path(path).open(newline='') as handle:
            for row in csv.DictReader(handle):
                participant, sid = row['participant_id'], row['summary_id']
                pair = (participant, sid)
                if participant not in {f'P{i:02d}' for i in range(1, 11)} or sid not in key or pair in seen:
                    raise ValueError(f'Unknown or duplicate rating: {pair}')
                label = row['judgment'].strip().lower()
                if label not in LABELS:
                    raise ValueError(f'Missing/invalid judgment: {pair}')
                if label == 'inconsistent' and not row['evidence'].strip():
                    raise ValueError(f'Inconsistent judgment needs source-based evidence: {pair}')
                seen.add(pair)
                votes[sid][label] += 1
    if len(seen) != 200:
        raise ValueError(f'Expected 10 participants × 20 summaries; received {len(seen)} ratings')
    model_votes = defaultdict(Counter)
    for sid, counts in votes.items():
        model_votes[key[sid]['mode']].update(counts)
    overall = sum(votes.values(), Counter())
    agreement = sum(sum(n * (n - 1) for n in c.values()) / 90 for c in votes.values()) / 20
    chance = sum((overall[label] / 200) ** 2 for label in LABELS)
    return {'participants': 10, 'summaries': 20, 'distinct_articles': len({v['example_id'] for v in key.values()}),
            'ratings': 200, 'per_summary': dict(votes),
            'per_model': {mode: {'counts': dict(counts), 'consistent_fraction_among_decided':
                counts['consistent'] / (counts['consistent'] + counts['inconsistent'])
                if counts['consistent'] + counts['inconsistent'] else None}
                for mode, counts in sorted(model_votes.items())},
            'pairwise_agreement': agreement,
            'fleiss_kappa_three_categories': (agreement - chance) / (1 - chance) if chance < 1 else None,
            'limitation': 'Descriptive study: five shared articles, one training seed. Ratings and summaries '
                          'share participants/articles; 200 ratings are not 200 independent examples.'}

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, default=root / 'responses')
    parser.add_argument('--key', type=Path, default=root / 'summary_key.json')
    parser.add_argument('--output', type=Path, default=root / 'analysis.json')
    args = parser.parse_args()
    files = [args.responses / f'p{i:02d}.csv' for i in range(1, 11)]
    result = analyze(args.key, files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Validated {result["ratings"]} ratings; saved {args.output}')
