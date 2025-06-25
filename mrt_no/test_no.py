import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

SEQ_LEN = 30
CHECKPOINT_DIR = './checkpoints_v3'

class MetroDataset(Dataset):
    def __init__(self, series, seq_len=30):
        self.seq_len = seq_len
        self.series = series.values

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len]
        y = self.series[idx+self.seq_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

def load_data():
    df = pd.read_csv('features_ready.csv')
    df = df.dropna()
    series = df['總運量']
    split_idx = int(len(series) * 0.8)
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]
    mean = train_series.mean()
    std = train_series.std()
    test_series = (test_series - mean) / std
    
    return test_series, mean, std

def load_checkpoint_for_epoch(model, epoch):
    path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint for epoch {epoch}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint: checkpoint_epoch_{epoch}.pt")
    return model

def evaluate(model, dataset, mean, std):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).item()
            preds.append(pred)
            trues.append(y.item())
    preds = np.array(preds) * std + mean
    trues = np.array(trues) * std + mean
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    mape = np.mean(np.abs((preds - trues) / trues)) * 100
    return mae, rmse, mape

def main():
    test_series, mean, std = load_data()
    test_dataset = MetroDataset(test_series, seq_len=SEQ_LEN)
    model = RNNModel()

    epochs = list(range(10, 901, 10))
    valid_epochs = []
    maes, rmses, mapes = [], [], []

    for epoch in epochs:
        try:
            model = load_checkpoint_for_epoch(model, epoch)
            mae, rmse, mape = evaluate(model, test_dataset, mean, std)
            valid_epochs.append(epoch)
            maes.append(mae)
            rmses.append(rmse)
            mapes.append(mape)
            print(f"Epoch {epoch} | MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        except FileNotFoundError as e:
            print(e)

    # 繪圖
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(valid_epochs, maes, label='MAE')
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    plt.subplot(1, 3, 2)
    plt.plot(valid_epochs, rmses, label='RMSE', color='orange')
    plt.title("RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")

    plt.subplot(1, 3, 3)
    plt.plot(valid_epochs, mapes, label='MAPE', color='green')
    plt.title("MAPE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAPE (%)")

    plt.tight_layout()
    plt.savefig("test_total_only_metrics2.png")
    print("📈 圖表已儲存：test_total_only_metrics.png")

if __name__ == '__main__':
    main()