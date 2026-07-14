import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


class AdversarialRobustnessEvaluator:
    def __init__(self, model, feature_groups):
        self.model = model
        self.feature_groups = feature_groups
        self.budgets = [0.0, 0.05, 0.10, 0.15, 0.20]

    def apply_perturbation(self, X, epsilon):
        noise = np.random.uniform(-epsilon, epsilon, X.shape)
        return X + noise

    def evaluate_resilience(self, X_test, y_true):
        results = {}
        for eps in self.budgets:
            X_adv = self.apply_perturbation(X_test, eps)
            X_tensor = torch.FloatTensor(X_adv)

            with torch.no_grad():
                logits = self.model(X_tensor)
                y_pred = logits.argmax(axis=1).numpy()

            results[eps] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return results

    def analyze_feature_vulnerability(self, X_test, y_true, epsilon=0.10):
        baseline_f1 = f1_score(y_true, self.model(torch.FloatTensor(X_test)).argmax(axis=1).numpy(), average='macro')
        vulnerability = {}

        for group_name, cols in self.feature_groups.items():
            X_adv = X_test.copy()
            noise = np.random.uniform(-epsilon, epsilon, X_adv[:, cols].shape)
            X_adv[:, cols] += noise

            with torch.no_grad():
                f1_adv = f1_score(y_true, self.model(torch.FloatTensor(X_adv)).argmax(axis=1).numpy(), average='macro')

            vulnerability[group_name] = (baseline_f1 - f1_adv) * 100
        return vulnerability

    def plot_figure14(self, robust_results, save_path='figure_14_adversarial.pdf'):
        budgets = list(robust_results.keys())
        f1_scores = list(robust_results.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(budgets, f1_scores, marker='o', linestyle='-', color='black', label='Macro-F1 Score')

        # 标注相对下降百分比
        baseline = f1_scores[0]
        for i in range(1, len(budgets)):
            degradation = (f1_scores[i] - baseline) / baseline * 100
            ax.text(budgets[i], f1_scores[i] + 0.005, f"{degradation:.1f}%", ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.5))

        ax.set_xlabel('Adversarial Perturbation Budget (ε)', fontsize=12)
        ax.set_ylabel('Macro-F1 Score', fontsize=12)
        ax.set_ylim(0.70, 0.90)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)
        print(f"Robustness plot saved to: {save_path}")

# 使用逻辑 (Integration Guide):
# feature_groups 定义示例:
# groups = {
#    'Temporal Entropy': [23, 24],
#    'Graph Topology': [15, 16],
#    'Transaction Value': [2, 3, 4, 5]
# }
# evaluator = AdversarialRobustnessEvaluator(model, groups)
# robust_results = evaluator.evaluate_resilience(X_test_scaled, y_test)
# vulnerability = evaluator.analyze_feature_vulnerability(X_test_scaled, y_test)