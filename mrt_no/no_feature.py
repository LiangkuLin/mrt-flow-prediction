

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 1000
SAVE_EVERY = 10
CHECKPOINT_DIR = './checkpoints_v3'  # 使用新目錄
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class MetroDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.seq_len = seq_len
        self.features = data['總運量'].values.reshape(-1, 1)  # 僅使用總運量，形狀 [n, 1]
        self.targets = data['總運量'].values

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]  # [seq_len, 1]
        y = self.targets[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

def load_data(seq_len=30):
    df = pd.read_csv('features_ready.csv')
    df = df.dropna()

    # 僅選擇總運量
    df = df[['總運量']]

    # Split 80% train, 20% test with time order preserved
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Normalize using only train set statistics
    mean = train_df.mean()
    std = train_df.std()
    std = std.replace(0, 1)  # 避免除零

    train_df = (train_df - mean) / std
    test_df = (test_df - mean) / std

    # 保存均值和標準差
    joblib.dump(mean, os.path.join(CHECKPOINT_DIR, 'mean.pkl'))
    joblib.dump(std, os.path.join(CHECKPOINT_DIR, 'std.pkl'))

    return train_df, test_df, mean, std

def save_checkpoint(model, optimizer, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt'))

def load_checkpoint(model, optimizer):
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
    if checkpoints:
        latest = checkpoints[-1]
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    return 0

def train():
    train_df, test_df, mean, std = load_data(seq_len=SEQ_LEN)
    train_dataset = MetroDataset(train_df, seq_len=SEQ_LEN)
    test_dataset = MetroDataset(test_df, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = RNNModel(input_size=1)  # input_size=1（僅總運量）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    start_epoch = load_checkpoint(model, optimizer)  # 修正參數
    print(f'Resuming from epoch {start_epoch}' if start_epoch > 0 else 'Starting from scratch')

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        # 訓練
        model.train()
        train_loss = 0
        for x, y in train_loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # 測試
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x)
                test_loss += loss_fn(pred, y).item()
        avg_test_loss = test_loss / len(test_loader)

        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)

if __name__ == '__main__':
    train()