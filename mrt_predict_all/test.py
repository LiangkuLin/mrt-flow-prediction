import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


SEQ_LEN = 30
CHECKPOINT_DIR = 'checkpoints_AttnRNNModel_10'

class MetroDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.seq_len = seq_len
        self.features = data.drop('總運量', axis=1).values
        self.targets = data['總運量'].values

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
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
    
class AttnRNNModel(nn.Module):
    """
    改良版 LSTM：
    - 多層、雙向
    - LayerNorm
    - 時間步注意力 (temporal attention)
    - MLP head
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_size  = hidden_size
        self.num_directions = 2 if bidirectional else 1

        # ① LSTM 本體
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # 官方建議
            bidirectional=bidirectional,
            batch_first=True
        )

        # ② LayerNorm（放在 attention 前，效果通常比直接在 LSTM 輸出後更穩）
        self.norm = nn.LayerNorm(hidden_size * self.num_directions)

        # ③ 時間步注意力：簡單線性層計分 → softmax
        self.attn_score = nn.Linear(
            hidden_size * self.num_directions,
            1,
            bias=False
        )

        # ④ MLP Head
        self.head = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        # LSTM
        out, _ = self.rnn(x)                       # (B, T, H*num_dir)

        # LayerNorm
        out = self.norm(out)

        # Attention：算每個時間步的權重
        #   score: (B, T, 1) → squeeze → (B, T)
        score = self.attn_score(out).squeeze(-1)
        alpha = F.softmax(score, dim=1)            # (B, T)

        # 加權和得到 context 向量
        context = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (B, H*num_dir)

        # MLP 預測
        y_hat = self.head(context).squeeze(-1)     # (B,)

        return y_hat


def load_data():
    df = pd.read_csv('features_ready.csv')

    # 1. 類別 → 數值
    df['是否為節日'] = df['是否為節日'].map({'是': 1, '否': 0})

    # 2. 星期 one-hot
    weekday_oh = pd.get_dummies(df['星期幾'], prefix='weekday', dtype=np.float32)
    df = pd.concat([df.drop(columns=['星期幾']), weekday_oh], axis=1)

    # 3-a 月份週期化
    df['month_sin'] = np.sin(2 * np.pi * (df['月份'] - 1) / 12).astype(np.float32)
    df['month_cos'] = np.cos(2 * np.pi * (df['月份'] - 1) / 12).astype(np.float32)
    df = df.drop(columns=['月份'])

    # 3-b 「日」週期化（1‒31 → 0‒30）
    df['day_sin'] = np.sin(2 * np.pi * (df['日'] - 1) / 31).astype(np.float32)
    df['day_cos'] = np.cos(2 * np.pi * (df['日'] - 1) / 31).astype(np.float32)
    df = df.drop(columns=['日'])        # 不再需要原始整數「日」

    # 4. 欄位順序（先連續／二元，再 sin/cos，再 one-hot）
    base_cols = [
        '總運量', '是否週末', '是否寒暑假',
        '是否連假前一日', '是否連假最後日', '是否為節日',
        '滾動平均_7日', '滾動平均_14日', '去年同日運量',
        'month_sin', 'month_cos',
        'day_sin',   'day_cos'
    ]
    df = df[base_cols + list(weekday_oh.columns)]
    df = df.dropna()

    # 5. Train / Test split
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # 6. 只標準化連續欄位（保持 0/1 與 sin/cos [-1,1] 不動）
    continuous_cols = [
        '總運量', '滾動平均_7日', '滾動平均_14日', '去年同日運量'
    ]
    mean = train_df[continuous_cols].mean()
    std  = train_df[continuous_cols].std().replace(0, 1)

    train_df[continuous_cols] = (train_df[continuous_cols] - mean) / std
    test_df[continuous_cols]  = (test_df[continuous_cols]  - mean) / std

    # 7. 統一 float32
    train_df = train_df.astype('float32')
    test_df  = test_df.astype('float32')

    return train_df, test_df, mean, std


def load_checkpoint(model, checkpoint_name):
    path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found in {CHECKPOINT_DIR}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint: {checkpoint_name}")
    return model

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

    preds = np.array(preds)
    trues = np.array(trues)
    mse = np.mean((preds - trues)**2)
    preds = preds * std['總運量'] + mean['總運量']
    trues = trues * std['總運量'] + mean['總運量']
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    mape = np.mean(np.abs((preds - trues) / trues)) * 100
    print(f"Test MSE: {mse:.2f}")
    print(f"Test MAE: {mae:.2f} 人")
    print(f"Test RMSE: {rmse:.2f} 人")
    print(f"Test MAPE: {mape:.2f}%")
    return mae, rmse, mape

def plot_metrics(epochs, metrics):
    plt.figure(figsize=(15, 5))
    
    # Plot MAE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, metrics['mae'], 'b-', label='MAE')
    plt.title('Test MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.grid(True)
    
    # Plot RMSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, metrics['rmse'], 'r-', label='RMSE')
    plt.title('Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # Plot MAPE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, metrics['mape'], 'g-', label='MAPE')
    plt.title('Test MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_metrics.png')
    plt.close()

def main():
    epochs = list(range(10, 801, 10))
    
    train_df, test_df, mean, std = load_data()
    test_dataset = MetroDataset(test_df, seq_len=SEQ_LEN)
    model = RNNModel(input_size=test_dataset[0][0].shape[1])
    # model = AttnRNNModel(
    #     input_size=test_dataset[0][0].shape[1],   # 你的特徵數
    #     hidden_size=64,
    #     num_layers=1,
    #     dropout=0.1,
    #     bidirectional=False
    # )
    
    # 收集所有checkpoint的評估結果
    metrics = {'mae': [], 'rmse': [], 'mape': []}
    
    for epoch in epochs:
        try:
            model = load_checkpoint_for_epoch(model, epoch)
            
            print(f"\n--- 評估 {epoch} ---")
            mae, rmse, mape = evaluate(model, test_dataset, mean, std)
            
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['mape'].append(mape)
            
        except FileNotFoundError:
            print(f"找不到checkpoint: {epoch}")
            continue
    
    # 繪製評估指標圖表
    plot_metrics(epochs, metrics)

if __name__ == '__main__':
    main()
