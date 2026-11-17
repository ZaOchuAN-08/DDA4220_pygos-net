"""Regenerate compact summary tables and curves from the committed records."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = [('baseline', 'Baseline', '#087e8b'), ('dcb', '+DCB', '#d26836')]


def main():
    summary = []
    tiers = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout='constrained')
    for folder, label, color in RUNS:
        data = json.loads((ROOT / folder / 'metrics.json').read_text())
        overall = data['overall']
        summary.append({'model': label, 'epoch': 20, 'split': 'val',
                        **{k: overall[k] for k in ['n', 'mae', 'rmse', 'relative_count_error', 'gt_total', 'pred_total']}})
        for tier in data['tiers']:
            tiers.append({'model': label, **{k: tier[k] for k in ['tier', 'n', 'mae', 'rmse']}})
        with (ROOT / folder / 'history.csv').open() as f:
            history = list(csv.DictReader(f))
        for ax, key, title in zip(axes, ['val_mae', 'val_rmse'], ['Validation MAE', 'Validation RMSE']):
            ax.plot([int(r['epoch']) for r in history], [float(r[key]) for r in history],
                    label=label, color=color, linewidth=1.8)
            ax.set(title=title, xlabel='Epoch', ylabel='Penguin count error')
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.grid(alpha=.2)
            ax.legend(frameon=False)
    fig.suptitle('Preliminary 20-epoch runs | final-epoch comparison, single seed')
    fig.savefig(ROOT / 'validation_curves.png', dpi=180)
    plt.close(fig)
    for name, rows in [('comparison.csv', summary), ('tier_comparison.csv', tiers)]:
        with (ROOT / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
