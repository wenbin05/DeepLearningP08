"""Plot saved multi-seed summaries without hard-coded experimental values."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.comparison import validate_comparison


def generate_pareto_plot(summary_path, output_path, dataset='cnn_dailymail', phase='test'):
    data = [r for r in json.loads(Path(summary_path).read_text())
            if r['dataset'] == dataset and r['phase'] == phase]
    if not data:
        raise ValueError('No results for the requested dataset and phase')
    validate_comparison(data)
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in data:
        label = f"{row['placement']} r={row['rank']} {row['projections']} (n={row['n']})"
        score = row['metrics']['rouge1']
        ax.errorbar(row['parameters']['trainable'], score['mean'], yerr=score['std'],
                    fmt='o', capsize=4, label=label)
    ax.set_xscale('symlog', linthresh=1)  # Includes zero-shot's zero trainable parameters.
    title = (f'{dataset} {phase}: single-seed scores (uncertainty not estimated)'
             if all(r['n'] == 1 for r in data)
             else f'{dataset} {phase}: mean ± sample standard deviation')
    ax.set(xlabel='Trainable parameters (symlog)', ylabel='ROUGE-1 F1 × 100', title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--output', default='figures/pareto_tradeoff.png')
    p.add_argument('--dataset', choices=['cnn_dailymail', 'xsum'], default='cnn_dailymail')
    p.add_argument('--phase', choices=['validation', 'test'], default='test')
    a = p.parse_args()
    generate_pareto_plot(a.summary, a.output, a.dataset, a.phase)
