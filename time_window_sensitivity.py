import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TimeWindowEvaluator:
    def __init__(self, windows=['1 hour', '6 hours', '1 day', '3 days', '1 week']):
        self.windows = windows

    def compute_metrics(self, y_true_dict, y_prob_dict):
        results = {}
        for w in self.windows:
            acc_runs, f1_runs, auc_runs = [], [], []

            for y_t, y_p in zip(y_true_dict[w], y_prob_dict[w]):
                y_pred_labels = y_p.argmax(axis=1)

                acc_runs.append(accuracy_score(y_t, y_pred_labels))
                f1_runs.append(f1_score(y_t, y_pred_labels, average='macro', zero_division=0))

                try:
                    auc_runs.append(roc_auc_score(y_t, y_p, average='macro', multi_class='ovr'))
                except ValueError:
                    auc_runs.append(np.nan)

            results[w] = {
                'Macro-F1_mean': np.mean(f1_runs), 'Macro-F1_std': np.std(f1_runs),
                'Accuracy_mean': np.mean(acc_runs), 'Accuracy_std': np.std(acc_runs),
                'AUC_mean': np.nanmean(auc_runs), 'AUC_std': np.nanstd(auc_runs)
            }
        return results

    def plot_figure11(self, results, save_path='figure_11_time_window.pdf'):
        x = np.arange(len(self.windows))

        f1_means = [results[w]['Macro-F1_mean'] for w in self.windows]
        f1_stds = [results[w]['Macro-F1_std'] for w in self.windows]
        acc_means = [results[w]['Accuracy_mean'] for w in self.windows]
        acc_stds = [results[w]['Accuracy_std'] for w in self.windows]
        auc_means = [results[w]['AUC_mean'] for w in self.windows]
        auc_stds = [results[w]['AUC_std'] for w in self.windows]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.errorbar(x, f1_means, yerr=f1_stds, fmt='-ko', label='Macro-F1', capsize=5, linewidth=2)
        ax2.errorbar(x, acc_means, yerr=acc_stds, fmt='--s', color='gray', label='Accuracy', capsize=5)
        ax2.errorbar(x, auc_means, yerr=auc_stds, fmt='-.^', color='lightgray', label='AUC', capsize=5)

        ax1.set_xlabel('Aggregation Time Window', fontsize=12)
        ax1.set_ylabel('Macro-F1 Score', fontsize=12)
        ax2.set_ylabel('Accuracy / AUC', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.windows, fontsize=11)

        opt_idx = np.argmax(f1_means)
        ax1.annotate('Optimal Resolution\n(1 day)',
                     xy=(x[opt_idx], f1_means[opt_idx]),
                     xytext=(x[opt_idx], f1_means[opt_idx] + max(f1_stds) + 0.01),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     ha='center', va='bottom', fontweight='bold')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=11)

        ax1.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Time-window sensitivity plot successfully saved to: {save_path}")


def main():
    windows = ['1 hour', '6 hours', '1 day', '3 days', '1 week']
    n_runs = 10

    y_true_dict = {w: [] for w in windows}
    y_prob_dict = {w: [] for w in windows}

    # =====================================================================
    # 操作指南：加载真实的实验数据
    # 请遍历读取您在 5 个不同时间窗下跑出的 10 次独立实验结果（.npy 文件）。
    # 注意：AUC 需要用到预测概率，所以请保存 torch.exp(out) 而不仅仅是 argmax 标签。
    # =====================================================================
    """
    for w in windows:
        folder_name = w.replace(' ', '_')
        for i in range(n_runs):
            y_true_dict[w].append(np.load(f'./results/time_windows/{folder_name}/run_{i}_y_true.npy'))
            y_prob_dict[w].append(np.load(f'./results/time_windows/{folder_name}/run_{i}_y_prob.npy'))

    evaluator = TimeWindowEvaluator(windows)
    results = evaluator.compute_metrics(y_true_dict, y_prob_dict)
    evaluator.plot_figure11(results)
    """


if __name__ == "__main__":
    main()