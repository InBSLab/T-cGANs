import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import logging
from scipy.stats import entropy
import psutil
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import random
import string
import datetime
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATA_PATH = 'D:\\FirstKBS\\Experiment\\venv\\Data_clean\\cleaned_ethereum_transactions2.csv'
FEATURES_PATH = "account_features.csv"
CGAN_DATA_PATHS = {
    "gambling": "generated_gambling_features.npy",
    "honeypot": "generated_honeypot_features.npy",
    "ponzi": "generated_ponzi_features.npy"
}
CHUNK_SIZE = 10000

logging.basicConfig(filename='feature_engineering.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    logging.info(f"Current memory usage: {mem:.2f} MB")


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def read_data(file_path):
    try:
        chunks = pd.read_csv(file_path, chunksize=CHUNK_SIZE, dtype={
            'isError': 'object', 'toCreate': 'object', 'timestamp': 'object',
            'from': 'object', 'to': 'object', 'value': 'float64',
            'gasLimit': 'float64', 'gasPrice': 'float64', 'gasUsed': 'float64'
        }, low_memory=False)

        df_list = []
        for chunk in chunks:
            df_list.append(chunk)

        df = pd.concat(df_list, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year

        return df
    except Exception as e:
        logging.error(f"error {str(e)}")
        raise


def get_account_lists(df):
    try:
        account_types = {
            'Exchange': 'exchange_addresses', 'Service': 'service_addresses',
            'Phishing': 'phishing_addresses', 'Bot': 'bot_addresses',
            'ICO Wallets': 'ico_addresses', 'Mining': 'mining_addresses',
            'Yield Farming': 'farm_addresses', 'Honeypot': 'honeypot_addresses',
            'Ponzi': 'ponzi_addresses', 'Gambling': 'gambling_addresses'
        }

        addresses = {name: set() for name in account_types.values()}

        if 'fromType' in df.columns:
            for account_type, address_list_name in account_types.items():
                addresses[address_list_name].update(df[df['fromType'] == account_type]['from'].unique())
        if 'toType' in df.columns:
            for account_type, address_list_name in account_types.items():
                addresses[address_list_name].update(df[df['toType'] == account_type]['to'].unique())

        all_relevant_addresses = set().union(*addresses.values())
        return all_relevant_addresses, {k: list(v) for k, v in addresses.items()}
    except Exception as e:
        logging.error(f"error {str(e)}")
        raise


def calculate_features(df, relevant_addresses):
    try:
        all_addresses = pd.concat([df['from'], df['to']]).unique()
        features = pd.DataFrame(index=all_addresses)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        time_range = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400

        from_tx_counts = df.groupby('from').size()
        to_tx_counts = df.groupby('to').size()
        features['daily_from_tx_frequency'] = from_tx_counts / time_range
        features['daily_to_tx_frequency'] = to_tx_counts / time_range

        features['unique_from_counterparties'] = df.groupby('from')['to'].nunique()
        features['unique_to_counterparties'] = df.groupby('to')['from'].nunique()

        for prefix, group in [('from', 'from'), ('to', 'to')]:
            features[f'{prefix}_avg_tx_value'] = df.groupby(group)['value'].mean()
            features[f'{prefix}_max_tx_value'] = df.groupby(group)['value'].max()
            features[f'{prefix}_min_tx_value'] = df.groupby(group)['value'].min()
            features[f'{prefix}_tx_value_std'] = df.groupby(group)['value'].std()

        incoming = df.groupby('to')['value'].sum()
        outgoing = df.groupby('from')['value'].sum()
        features['in_out_ratio'] = incoming.div(outgoing.add(1e-10)).fillna(0)

        for metric in ['gasUsed', 'gasPrice']:
            features[f'avg_{metric}'] = df.groupby('from')[metric].mean()
            features[f'max_{metric}'] = df.groupby('from')[metric].max()
            features[f'{metric}_std'] = df.groupby('from')[metric].std()

        for prefix, group in [('from', 'from'), ('to', 'to')]:
            first_tx = df.groupby(group)['timestamp'].min()
            last_tx = df.groupby(group)['timestamp'].max()
            features[f'{prefix}_account_active_time'] = (last_tx - first_tx).dt.total_seconds() / 86400

        def calculate_entropy(x):
            hour_counts = x['timestamp'].dt.hour.value_counts()
            probabilities = hour_counts / len(x)
            return entropy(probabilities)

        features['from_tx_time_entropy'] = df.groupby('from').apply(calculate_entropy)
        features['to_tx_time_entropy'] = df.groupby('to').apply(calculate_entropy)

        error_tx = df[df['isError'] == 1].groupby('from').size()
        features['tx_failure_rate'] = error_tx.div(from_tx_counts).fillna(0)

        contract_creation = df[df['toCreate'] == 1].groupby('from').size()
        features['contract_creation_rate'] = contract_creation.div(from_tx_counts).fillna(0)

        def avg_interval(x):
            if len(x) < 2: return 0
            return x.sort_values().diff().dt.total_seconds().mean()

        def std_interval(x):
            if len(x) < 2: return 0
            return x.sort_values().diff().dt.total_seconds().std()

        for prefix, group in [('from', 'from'), ('to', 'to')]:
            features[f'{prefix}_avg_tx_interval'] = df.groupby(group)['timestamp'].apply(avg_interval)
            features[f'{prefix}_tx_interval_std'] = df.groupby(group)['timestamp'].apply(std_interval)

        max_timestamp = df['timestamp'].max()
        features['from_days_since_first_tx'] = (max_timestamp - df.groupby('from')[
            'timestamp'].min()).dt.total_seconds() / 86400
        features['to_days_since_first_tx'] = (max_timestamp - df.groupby('to')[
            'timestamp'].min()).dt.total_seconds() / 86400

        for col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        features = features.fillna(0)

        return features
    except Exception as e:
        logging.error(f"error: {str(e)}")
        raise


class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim, hidden_dim=260, num_layers=3, dropout=0.5):
        super(Generator, self).__init__()
        self.model = nn.Sequential()
        input_dim = noise_dim + condition_dim

        for i in range(num_layers):
            self.model.add_module(f'linear_{i}', nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.model.add_module(f'bn_{i}', nn.BatchNorm1d(hidden_dim))
            self.model.add_module(f'lrelu_{i}', nn.LeakyReLU(0.2))
            self.model.add_module(f'dropout_{i}', nn.Dropout(dropout))

        self.model.add_module('output', nn.Linear(hidden_dim, output_dim))
        self.model.add_module('tanh', nn.Tanh())

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim=180, num_layers=3, dropout=0.5):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential()
        total_input_dim = input_dim + condition_dim

        for i in range(num_layers):
            self.model.add_module(f'linear_{i}', nn.Linear(total_input_dim if i == 0 else hidden_dim, hidden_dim))
            self.model.add_module(f'lrelu_{i}', nn.LeakyReLU(0.2))
            self.model.add_module(f'dropout_{i}', nn.Dropout(dropout))

        self.model.add_module('output', nn.Linear(hidden_dim, 1))

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)


class AccountDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    node_features = pd.read_csv(FEATURES_PATH, index_col=None)

    if node_features.columns[0] != 'address':
        node_features.rename(columns={node_features.columns[0]: 'address'}, inplace=True)

    node_features = node_features.dropna(subset=['address'])
    node_features.set_index('address', inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_range = (df['timestamp'].min(), df['timestamp'].max())
    return df, node_features, time_range


def load_cgan_data():
    cgan_data = {}
    for account_type, file_path in CGAN_DATA_PATHS.items():
        try:
            data = np.load(file_path)
            cgan_data[account_type] = data
        except Exception as e:
            print(f"Error loading CGAN data for {account_type}: {e}")
    return cgan_data


def preprocess_cgan_data(cgan_data, original_features):
    preprocessed_data = {}
    label_mapping = {'gambling': 9, 'honeypot': 7, 'ponzi': 8}
    feature_columns = original_features.columns.drop('label')

    for account_type, features in cgan_data.items():
        cgan_df = pd.DataFrame(features, columns=feature_columns)
        cgan_df['label'] = label_mapping[account_type]
        cgan_df['address'] = [f"cgan_{account_type}_{i}" for i in range(len(cgan_df))]
        cgan_df.set_index('address', inplace=True)
        preprocessed_data[account_type] = cgan_df
    return preprocessed_data


def merge_datasets(original_data, cgan_data):
    all_columns = set(original_data.columns)
    for df in cgan_data.values():
        all_columns.update(df.columns)

    for df in [original_data] + list(cgan_data.values()):
        for col in all_columns:
            if col not in df.columns:
                df[col] = np.nan

    service_accounts = original_data[original_data['label'] == 1]
    other_accounts = original_data[original_data['label'] != 1]

    if len(service_accounts) > 519:
        service_accounts = service_accounts.sample(n=519, random_state=42)

    combined_data = pd.concat([other_accounts, service_accounts] + list(cgan_data.values()), axis=0)
    combined_data = combined_data.reset_index()
    combined_data.rename(columns={'index': 'address'}, inplace=True)
    combined_data['address'] = combined_data['address'].astype(str)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_data['label'] = combined_data['label'].astype(int)

    return combined_data


def extract_features(combined_data):
    numeric_features = combined_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['label', 'address']
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    scaler = StandardScaler()
    combined_data[numeric_features] = scaler.fit_transform(combined_data[numeric_features])
    combined_data = combined_data.fillna(0)
    return combined_data, numeric_features


def build_graph(df, combined_node_data):
    G = nx.DiGraph()
    address_to_index = {addr: idx for idx, addr in enumerate(combined_node_data['address'])}

    for idx, row in combined_node_data.iterrows():
        features = row.drop(['address', 'label']).values
        feature_dict = {f'feature_{i}': float(v) for i, v in enumerate(features)}
        G.add_node(idx, **feature_dict, label=int(row['label']), address=row['address'])

    edge_data = []
    for _, row in df.iterrows():
        from_idx = address_to_index.get(row['from'])
        to_idx = address_to_index.get(row['to'])
        if from_idx is not None and to_idx is not None:
            timestamp = row['timestamp'].timestamp()
            value = float(row['value'])
            G.add_edge(from_idx, to_idx, timestamp=timestamp, value=value)
            edge_data.append((from_idx, to_idx, timestamp, value))

    edge_index = torch.tensor([[e[0], e[1]] for e in edge_data], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[e[2], e[3]] for e in edge_data], dtype=torch.float)
    node_features = torch.tensor([list(G.nodes[n].values())[:-2] for n in range(len(G.nodes()))], dtype=torch.float)

    label_to_index = {label: idx for idx, label in enumerate(combined_node_data['label'].unique())}
    y = torch.tensor([label_to_index[G.nodes[n]['label']] for n in range(len(G.nodes()))], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data, G, label_to_index


def load_and_process_data():
    df, node_features, time_range = load_data()
    cgan_data = load_cgan_data()
    preprocessed_cgan_data = preprocess_cgan_data(cgan_data, node_features)
    combined_node_data = merge_datasets(node_features, preprocessed_cgan_data)
    combined_node_data, numeric_features = extract_features(combined_node_data)
    data, G, label_to_index = build_graph(df, combined_node_data)
    # 额外返回 df 以供严格的时序划分使用
    return data, G, combined_node_data, label_to_index, df


def create_chronological_split_and_mask(df, combined_node_data, data, cgan_prefix='cgan_'):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    first_tx_from = df.groupby('from')['timestamp'].min()
    first_tx_to = df.groupby('to')['timestamp'].min()
    first_tx_combined = pd.concat([first_tx_from, first_tx_to]).groupby(level=0).min()

    timestamps = []
    is_cgan = []
    for addr in combined_node_data['address']:
        if str(addr).startswith(cgan_prefix):
            timestamps.append(pd.Timestamp.min)
            is_cgan.append(True)
        else:
            timestamps.append(first_tx_combined.get(addr, pd.Timestamp.max))
            is_cgan.append(False)

    combined_node_data['first_tx_ts'] = timestamps
    combined_node_data['is_cgan'] = is_cgan

    real_nodes = combined_node_data[~combined_node_data['is_cgan']].copy()
    cgan_nodes = combined_node_data[combined_node_data['is_cgan']].copy()
    real_nodes = real_nodes.sort_values('first_tx_ts')

    num_real = len(real_nodes)
    train_end = int(0.6 * num_real)
    val_end = int(0.7 * num_real)

    train_real = real_nodes.iloc[:train_end]
    val_real = real_nodes.iloc[train_end:val_end]
    test_real = real_nodes.iloc[val_end:]

    train_cutoff_ts = train_real['first_tx_ts'].max()
    train_cutoff_timestamp = train_cutoff_ts.timestamp()

    train_indices = torch.tensor(train_real.index.tolist() + cgan_nodes.index.tolist(), dtype=torch.long)
    val_indices = torch.tensor(val_real.index.tolist(), dtype=torch.long)
    test_indices = torch.tensor(test_real.index.tolist(), dtype=torch.long)

    edge_timestamps = data.edge_attr[:, 0]
    valid_edge_mask = edge_timestamps <= train_cutoff_timestamp
    masked_edge_index = data.edge_index[:, valid_edge_mask]
    masked_edge_attr = data.edge_attr[valid_edge_mask]

    return train_indices, val_indices, test_indices, masked_edge_index, masked_edge_attr


class TimeEncoder(nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncoder, self).__init__()
        self.basis_freq = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, expand_dim))).float())
        self.phase = nn.Parameter(torch.rand(expand_dim).float() * 2 * np.pi)

    def forward(self, ts):
        ts = ts.view(-1, 1)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        return torch.cos(map_ts)


class TGAT(nn.Module):
    def __init__(self, node_features, edge_features, num_classes, num_layers, num_heads, dropout, device):
        super(TGAT, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.final_dim = 32 * num_heads

        self.time_encoder = TimeEncoder(16)
        edge_dim = 16 + edge_features
        self.input_proj = nn.Linear(node_features, 32)

        self.attention_layers = nn.ModuleList([
            GATConv(32 if i == 0 else 32 * num_heads, 32, heads=num_heads, dropout=dropout, edge_dim=edge_dim,
                    add_self_loops=True)
            for i in range(num_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(32 * num_heads, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_proj(x)

        if edge_index.numel() == 0 or x.size(0) == 1:
            x = x.view(x.size(0), -1)
            x = F.pad(x, (0, self.final_dim - x.size(1)))
        else:
            time_encoding = self.time_encoder(edge_attr[:, 0].unsqueeze(1))
            edge_attr = torch.cat([time_encoding, edge_attr], dim=1)

            for i in range(self.num_layers):
                x = self.attention_layers[i](x, edge_index, edge_attr=edge_attr)
                if i < self.num_layers - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)

        x = self.mlp(x)
        return F.log_softmax(x, dim=-1)


class EarlyStopMonitor:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def train_tgat(model, train_loader, optimizer, criterion, device, accumulation_steps=2):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accumulation_steps
    return total_loss / len(train_loader)


def test_tgat(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(torch.exp(out).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred.argmax(axis=1)).mean()
    try:
        auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovr')
        ap = average_precision_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred.argmax(axis=1), average='macro')
    except ValueError as e:
        print(f"Error in calculating metrics: {e}")
        auc = ap = f1 = float('nan')

    return total_loss / len(loader), accuracy, auc, ap, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data, G, combined_node_data, label_to_index, df = load_and_process_data()

    combined_node_data = combined_node_data[combined_node_data['label'] != -1]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(combined_node_data['label'])
    num_classes = len(label_encoder.classes_)

    x = torch.FloatTensor(combined_node_data.drop(['label', 'address'], axis=1).values)
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    y = torch.LongTensor(y)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    print("Executing strictly chronological data splitting and temporal edge masking...")
    train_indices, val_indices, test_indices, masked_edge_index, masked_edge_attr = \
        create_chronological_split_and_mask(df, combined_node_data, data)

    data.edge_index = masked_edge_index
    data.edge_attr = masked_edge_attr

    train_loader = NeighborLoader(
        data,
        input_nodes=train_indices,
        num_neighbors=[5, 5],
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    val_loader = NeighborLoader(data, input_nodes=val_indices, num_neighbors=[10, 10], batch_size=64)
    test_loader = NeighborLoader(data, input_nodes=test_indices, num_neighbors=[10, 10], batch_size=64)

    node_features = data.x.shape[1]
    edge_features = data.edge_attr.shape[1]
    model = TGAT(node_features=node_features,
                 edge_features=edge_features,
                 num_classes=num_classes,
                 num_layers=3,
                 num_heads=8,
                 dropout=0.2,
                 device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.NLLLoss()
    early_stopping = EarlyStopMonitor(patience=15)

    num_epochs = 200
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train_tgat(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_ap, val_f1 = test_tgat(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        if early_stopping(val_loss):
            break

    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_auc, test_ap, test_f1 = test_tgat(model, test_loader, criterion, device)
    print(f'Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, '
          f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}, Test F1: {test_f1:.4f}')


if __name__ == "__main__":
    main()