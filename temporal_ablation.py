import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os


class TemporalAblationEvaluator:
    def __init__(self):
        self.configs = [
            'Full_System',
            'Remove_Timestamps',
            'Remove_Temporal_Encoding',
            'Remove_Both'
        ]

        self.display_names = [
            'Full System\n(All Temporal)',
            'Remove\nTimestamps\n(M\\T_stamp)',
            'Remove\nTemporal Encoding\n(M\\T_enc)',
            'Remove Both\n(M\\T)'
        ]

    def compute_metrics(self, y_true_dict, y_pred_dict):
        results = {}
        for config in self.configs:
            f1_runs = []
            for y_t, y_p in zip(y_true_dict[config], y_pred_dict[config]):
                y_pred_labels = y_p.argmax(axis=1) if len(y_p.shape) > 1 else y_p
                f1_runs.append(f1_score(y_t, y_pred_labels, average='macro', zero_division=0))

            results[config] = {
                'mean': np.mean(f1_runs),
                'std': np.std(f1_runs)
            }
        return results

    def plot_figure12(self, results, save_path='figure_12_temporal_ablation.pdf'):
        means = [results[c]['mean'] for c in self.configs]
        stds = [results[c]['std'] for c in self.configs]

        baseline_f1 = means[0]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.configs))
        width = 0.5

        patterns = ['-', '\\\\', '//', 'xx']
        colors = ['dimgray', 'darkgray', 'silver', 'lightgray']

        for i in range(len(self.configs)):
            ax.bar(x[i], means[i], width, yerr=stds[i],
                   color=colors[i], edgecolor='black', hatch=patterns[i], capsize=5)

            if i > 0:
                delta_pp = (means[i] - baseline_f1) * 100
                ax.text(x[i], means[i] + stds[i] + 0.005, f"{delta_pp:.1f} pp",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Macro-F1 Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(self.display_names, fontsize=11)

        min_y = max(0, min(means) - max(stds) - 0.05)
        ax.set_ylim(min_y, 0.90)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Temporal ablation plot successfully saved to: {save_path}")


def main():
    evaluator = TemporalAblationEvaluator()
    n_runs = 10

    y_true_dict = {config: [] for config in evaluator.configs}
    y_pred_dict = {config: [] for config in evaluator.configs}

    # =====================================================================
    # 操作指南：加载真实的实验数据
    # 请遍历读取您在 4 种时间配置下跑出的 10 次独立实验结果（.npy 文件）。
    # =====================================================================
    """
    for config in evaluator.configs:
        for i in range(n_runs):
            y_true_dict[config].append(np.load(f'./results/temporal/{config}/run_{i}_y_true.npy'))
            y_pred_dict[config].append(np.load(f'./results/temporal/{config}/run_{i}_y_pred.npy'))

    results = evaluator.compute_metrics(y_true_dict, y_pred_dict)
    evaluator.plot_figure12(results)
    """


if __name__ == "__main__":
    main()