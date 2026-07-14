import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ImbalancedLearningEvaluator:
    def __init__(self, class_names, minority_classes=['Gambling', 'Honeypot', 'Ponzi']):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.minority_classes = minority_classes
        self.minority_indices = [class_names.index(c) for c in minority_classes]

    def compute_table8_metrics(self, y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        w_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError:
            auc = np.nan

        return {
            'Accuracy': acc,
            'Macro-F1': macro_f1,
            'W-F1': w_f1,
            'AUC': auc,
            'MCC': mcc
        }

    def compute_figure9_minority_f1(self, y_true_runs, y_pred_runs):
        n_runs = len(y_true_runs)
        minority_f1_runs = {c: np.zeros(n_runs) for c in self.minority_classes}

        for i in range(n_runs):
            _, _, f1_all, _ = precision_recall_fscore_support(
                y_true_runs[i], y_pred_runs[i], labels=range(self.n_classes), zero_division=0
            )
            for c_name, c_idx in zip(self.minority_classes, self.minority_indices):
                minority_f1_runs[c_name][i] = f1_all[c_idx]

        results = {}
        for c_name in self.minority_classes:
            results[c_name] = {
                'mean': np.mean(minority_f1_runs[c_name]),
                'std': np.std(minority_f1_runs[c_name])
            }
        return results

    def plot_figure9(self, strategies_results, save_path='figure_9_minority_f1.pdf'):
        strategies = list(strategies_results.keys())
        x = np.arange(len(strategies))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, c_name in enumerate(self.minority_classes):
            means = [strategies_results[s][c_name]['mean'] for s in strategies]
            stds = [strategies_results[s][c_name]['std'] for s in strategies]

            offset = (idx - 1) * width
            ax.bar(x + offset, means, width, yerr=stds, label=c_name, capsize=5, edgecolor='black')

        ax.set_ylabel('F1 Score')
        ax.set_title('Performance Breakdown on Extreme Minority Classes')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(save_path, format='pdf', dpi=300)


def apply_feature_space_oversampling(X, y, strategy='SMOTE'):
    if strategy == 'SMOTE':
        sampler = SMOTE(random_state=42)
    elif strategy == 'Borderline-SMOTE':
        sampler = BorderlineSMOTE(random_state=42)
    elif strategy == 'ADASYN':
        sampler = ADASYN(random_state=42)
    else:
        return X, y

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


def get_cost_sensitive_weights(y, n_classes, device):
    class_counts = np.bincount(y, minlength=n_classes)
    total_samples = len(y)
    weights = total_samples / (n_classes * (class_counts + 1e-5))
    weights = torch.FloatTensor(weights).to(device)
    return weights


def get_loss_function(strategy, y_train=None, n_classes=10, device='cpu'):
    if strategy == 'Cost-sensitive':
        weights = get_cost_sensitive_weights(y_train, n_classes, device)
        return nn.CrossEntropyLoss(weight=weights)
    elif strategy == 'Focal Loss':
        weights = get_cost_sensitive_weights(y_train, n_classes, device)
        return FocalLoss(alpha=weights, gamma=2.0)
    else:
        return nn.CrossEntropyLoss()