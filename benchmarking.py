import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef


class SOTABenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.baselines = {
            'EvolveGCN': None,
            'ROLAND': None,
            'PEAE-GNN': None
        }

    def load_model(self, model_name, weights_path):
        # 此处根据您实际获取的 SOTA 模型权重路径进行加载
        # model = ...
        # model.load_state_dict(torch.load(weights_path))
        # self.baselines[model_name] = model.to(self.device)
        pass

    def evaluate(self, model_name, test_loader):
        model = self.baselines[model_name]
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                probs = torch.softmax(out, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(out.argmax(dim=1).cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

        y_prob = np.concatenate(all_probs)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)

        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Macro-F1': f1_score(y_true, y_pred, average='macro'),
            'W-F1': f1_score(y_true, y_pred, average='weighted'),
            'AUC': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }


def run_sota_comparison(test_loader):
    benchmark = SOTABenchmark()
    results = {}

    for model_name in benchmark.baselines.keys():
        print(f"Evaluating {model_name}...")
        results[model_name] = benchmark.evaluate(model_name, test_loader)

    return pd.DataFrame(results).T


if __name__ == "__main__":
    # 使用说明：
    # 1. 确保您的测试集 DataLoader 已经通过 chronological_split 处理 (防止时间泄漏)
    # 2. 将您获取的 2024 SOTA 模型权重路径放入 load_model 中
    # results_df = run_sota_comparison(test_loader)
    # print(results_df)
    # results_df.to_csv('sota_comparison_table5.csv')
    pass