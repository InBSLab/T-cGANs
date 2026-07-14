import torch
import pandas as pd
import numpy as np


def create_chronological_split_and_mask(df, combined_node_data, data, cgan_prefix='cgan_'):
    print("Initiating strict chronological validation protocol...")

    # 1. 获取每个真实账户的初始活动时间戳 (Initial activity timestamp)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    first_tx_from = df.groupby('from')['timestamp'].min()
    first_tx_to = df.groupby('to')['timestamp'].min()
    first_tx_combined = pd.concat([first_tx_from, first_tx_to]).groupby(level=0).min()

    # 2. 将时间戳映射到节点 DataFrame 中，并识别 cGANs 生成节点
    timestamps = []
    is_cgan = []
    for addr in combined_node_data['address']:
        if str(addr).startswith(cgan_prefix):
            # cGANs 节点没有真实初始时间，标记为最早时间以确保其仅进入训练集
            timestamps.append(pd.Timestamp.min)
            is_cgan.append(True)
        else:
            timestamps.append(first_tx_combined.get(addr, pd.Timestamp.max))
            is_cgan.append(False)

    combined_node_data['first_tx_ts'] = timestamps
    combined_node_data['is_cgan'] = is_cgan

    # 3. 分离真实节点与生成节点
    real_nodes = combined_node_data[~combined_node_data['is_cgan']].copy()
    cgan_nodes = combined_node_data[combined_node_data['is_cgan']].copy()

    # 4. 按初始时间戳对真实节点进行严格排序 (Chronological sorting)
    real_nodes = real_nodes.sort_values('first_tx_ts')

    # 5. 执行 60% (Train) - 10% (Val) - 30% (Test) 划分
    num_real = len(real_nodes)
    train_end = int(0.6 * num_real)
    val_end = int(0.7 * num_real)

    train_real = real_nodes.iloc[:train_end]
    val_real = real_nodes.iloc[train_end:val_end]
    test_real = real_nodes.iloc[val_end:]

    # 获取训练集截止时间戳 (Training cutoff timestamp)
    train_cutoff_ts = train_real['first_tx_ts'].max()
    train_cutoff_timestamp = train_cutoff_ts.timestamp()

    print(f"Chronological split completed. Training cutoff timestamp: {train_cutoff_ts}")

    # 6. 生成 PyTorch 索引，强制 cGANs 节点仅进入 Training Fold
    train_indices = torch.tensor(train_real.index.tolist() + cgan_nodes.index.tolist(), dtype=torch.long)
    val_indices = torch.tensor(val_real.index.tolist(), dtype=torch.long)
    test_indices = torch.tensor(test_real.index.tolist(), dtype=torch.long)

    # 7. 剔除未来交易边 (Masking future edges to prevent temporal leakage)
    # data.edge_attr[:, 0] 存储的是边的 timestamp
    edge_timestamps = data.edge_attr[:, 0]

    # 仅保留发生于 cutoff timestamp 之前的边
    valid_edge_mask = edge_timestamps <= train_cutoff_timestamp
    masked_edge_index = data.edge_index[:, valid_edge_mask]
    masked_edge_attr = data.edge_attr[valid_edge_mask]

    removed_edges = data.edge_index.shape[1] - masked_edge_index.shape[1]
    print(f"Temporal leakage prevention: Masked {removed_edges} future transactions.")

    return train_indices, val_indices, test_indices, masked_edge_index, masked_edge_attr