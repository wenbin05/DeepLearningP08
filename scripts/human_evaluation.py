"""Prepare blinded rating packets from genuine predictions; summarize real ratings."""
import argparse
import csv
import html
import json
from pathlib import Path
import random
from collections import Counter, defaultdict

from src.aggregate_results import aggregate
from src.comparison import validate_comparison
from src.train import write_json

MODES = {'zero_shot', 'encoder_self_attn', 'decoder_self_attn', 'cross_attn'}
LABELS = ('consistent', 'inconsistent', 'uncertain')
FIELDS = ['participant_id', 'summary_id', 'judgment', 'evidence']


def prepare(paths, output, seed=42):
    output = Path(output)
    records, result_paths = {}, []
    for path in map(Path, paths):
        result_path = path.with_name('test_results.json')
        result = json.loads(result_path.read_text())
        mode = result['manifest']['config']['placement']
        if mode in records or mode not in MODES or result['phase'] != 'test':
            raise ValueError('Provide one held-out prediction file for each isolated placement and zero-shot')
        if result['manifest']['seed'] != 42:
            raise ValueError('Human-study sampling protocol fixes training seed 42')
        # Check order, references and scores before sampling outputs.
        from scripts.check_final_results import rescore
        rescore(path, result)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [str(r['id']) for r in rows]
        if len(set(ids)) != len(ids) or any(r['id'] is None for r in rows):
            raise ValueError('Human sampling requires unique non-null example IDs')
        records[mode] = dict(zip(ids, rows))
        result_paths.append(result_path)
    if set(records) != MODES:
        raise ValueError('Exactly four configurations are required')
    validate_comparison(aggregate(result_paths, expected_seeds=(42,)))
    first = records['zero_shot']
    if len(first) < 5 or any(set(rows) != set(first) for rows in records.values()):
        raise ValueError('At least five identical source-example IDs are required')
    for key in first:
        if any((rows[key]['source'], rows[key]['reference_original']) !=
               (first[key]['source'], first[key]['reference_original']) for rows in records.values()):
            raise ValueError('Source/reference text differs between model outputs')
    rng = random.Random(seed)
    selected = rng.sample(sorted(first), 5)
    items = [(mode, key) for mode in sorted(MODES) for key in selected]
    rng.shuffle(items)
    key = {f'S{i+1:02d}': {'mode': mode, 'example_id': example}
           for i, (mode, example) in enumerate(items)}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'coordinator' / 'key.json', key)
    write_json(output / 'coordinator' / 'sampling.json', {
        'sampling_seed': seed, 'training_seed': 42, 'articles': selected,
        'summaries': 20, 'participants': 10, 'source_predictions': list(map(str, paths)),
        'design': 'Five shared articles × four configurations; each participant rates all 20 outputs'})
    for number in range(1, 11):
        participant = f'P{number:02d}'
        folder = output / 'participants' / participant
        folder.mkdir(parents=True)
        order = list(key)
        rng.shuffle(order)
        with (folder / 'ratings.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({'participant_id': participant, 'summary_id': sid,
                              'judgment': '', 'evidence': ''} for sid in order)
        sections = []
        for sid in order:
            item = key[sid]
            row = records[item['mode']][item['example_id']]
            sections.append(f'<section><h2>{sid}</h2><h3>Source article</h3><p>{html.escape(row["source"])}</p>'
                            f'<h3>Generated summary</h3><p>{html.escape(row["prediction"])}</p></section>')
        (folder / 'packet.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Factual consistency study</title>'
            '<style>body{max-width:900px;margin:2rem auto;font:18px/1.6 sans-serif}section{border-top:1px solid;padding:1rem 0}p{white-space:pre-wrap}</style>'
            f'<h1>Factual consistency — {participant}</h1><p>Compare each summary with its source article. '
            'In ratings.csv enter consistent (all factual claims supported), inconsistent (at least one unsupported '
            'or contradicted claim), or uncertain (cannot decide). Missing important information or poor style alone '
            'is not factual inconsistency. For inconsistent, quote the problematic claim and explain the source conflict '
            'in the evidence column. Do not enter your name or other personal details. Articles repeat across summaries; '
            'judge each summary independently. Participation is voluntary; follow your course guidance.</p>' + ''.join(sections))
    return key


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--predictions', nargs=4, type=Path, required=True)
    prep.add_argument('--output_dir', type=Path, required=True)
    analysis = commands.add_parser('analyze')
    analysis.add_argument('--key', type=Path, required=True)
    analysis.add_argument('--ratings', nargs='+', type=Path, required=True)
    analysis.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.predictions, args.output_dir)
    else:
        write_json(args.output, analyze(args.key, args.ratings))


if __name__ == '__main__':
    main()
