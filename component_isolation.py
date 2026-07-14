import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os


class ComponentIsolationEvaluator:
    def __init__(self, configs):
        self.configs = configs
        self.baseline_name = configs[0]

    def compute_macro_f1_stats(self, y_true_dict, y_pred_dict):
        results = {}
        for config in self.configs:
            f1_runs = []
            for y_t, y_p in zip(y_true_dict[config], y_pred_dict[config]):
                f1_runs.append(f1_score(y_t, y_p, average='macro', zero_division=0))
            results[config] = {
                'mean': np.mean(f1_runs),
                'std': np.std(f1_runs)
            }
        return results

    def generate_table7(self, results):
        baseline_f1 = results[self.baseline_name]['mean']
        table_data = []

        for config in self.configs:
            mean_f1 = results[config]['mean']
            std_f1 = results[config]['std']
            delta_pp = (mean_f1 - baseline_f1) * 100

            if config == self.baseline_name:
                delta_str = "-"
            else:
                delta_str = f"+{delta_pp:.1f} pp" if delta_pp > 0 else f"{delta_pp:.1f} pp"

            table_data.append({
                'Configuration': config,
                'Macro-F1': f"{mean_f1:.3f}±{std_f1:.3f}",
                'Δ from Baseline': delta_str
            })

        return pd.DataFrame(table_data)

    def plot_figure8(self, results, save_path='figure_8_component_isolation.pdf'):
        means = [results[c]['mean'] for c in self.configs]
        stds = [results[c]['std'] for c in self.configs]
        baseline_f1 = means[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(self.configs))
        width = 0.5

        patterns = ['//', '\\\\', 'xx', '..', '']
        colors = ['white', 'lightgray', 'silver', 'darkgray', 'dimgray']

        for i in range(len(self.configs)):
            ax.bar(x[i], means[i], width, yerr=stds[i],
                   color=colors[i], edgecolor='black', hatch=patterns[i], capsize=5)

            if i > 0:
                delta_pp = (means[i] - baseline_f1) * 100
                ax.text(x[i], means[i] + stds[i] + 0.015, f"+{delta_pp:.1f} pp",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Macro-F1 Score', fontsize=12)

        labels = [
            'Baseline\n(TGAT w/o cGANs)',
            'Weighted-loss\nTGAT',
            'cGANs +\nstatic GCN',
            'cGANs +\nstatic GAT',
            'Full System\n(TGAT + cGANs)'
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)


def main():
    configs = [
        'TGAT_Baseline',
        'Weighted_Loss_TGAT',
        'cGANs_Static_GCN',
        'cGANs_Static_GAT',
        'Full_System'
    ]

    n_runs = 10
    y_true_dict = {config: [] for config in configs}
    y_pred_dict = {config: [] for config in configs}

    # ---------------------------------------------------------------------
    # 请在此处加载您真实跑出的 5 个变体模型的预测结果 (.npy)
    # 例如：
    # for i in range(n_runs):
    #     for config in configs:
    #         y_true_dict[config].append(np.load(f'./results/{config}/run_{i}_y_true.npy'))
    #         y_pred_dict[config].append(np.load(f'./results/{config}/run_{i}_y_pred.npy'))
    # ---------------------------------------------------------------------

    # evaluator = ComponentIsolationEvaluator(configs)
    # results = evaluator.compute_macro_f1_stats(y_true_dict, y_pred_dict)

    # df_table7 = evaluator.generate_table7(results)
    # print("=== Table 7: Component Attribution and Isolation Analysis ===")
    # print(df_table7.to_string(index=False))

    # evaluator.plot_figure8(results)


if __name__ == "__main__":
    main()