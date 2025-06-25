import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 800
SAVE_EVERY = 10
CHECKPOINT_DIR = './checkpoints_AttnRNNModel_10'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    preds = np.array(preds) * std['總運量'] + mean['總運量']
    trues = np.array(trues) * std['總運量'] + mean['總運量']
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues)**2))
    mape = np.mean(np.abs((preds - trues) / trues)) * 100
    return mae, rmse, mape

def plot_metrics():
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
   
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def train():
    train_df, test_df, mean, std = load_data()
    # print(train_df[:2])
    train_dataset = MetroDataset(train_df, seq_len=SEQ_LEN)
    # print(train_dataset[0])
    # os._exit(0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RNNModel(input_size=train_dataset[0][0].shape[1])
    # model = AttnRNNModel(
    #     input_size=train_dataset[0][0].shape[1],   # 你的特徵數
    #     hidden_size=64,
    #     num_layers=1,
    #     dropout=0.1,
    #     bidirectional=False
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # start_epoch = load_checkpoint(model, optimizer)
    start_epoch = 0
    print(f'Resuming from epoch {start_epoch}' if start_epoch > 0 else 'Starting from scratch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 用於記錄訓練loss
    train_losses = []

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch}, Train Loss: {avg_loss:.6f}')
            
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch)
            # 繪製loss圖
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig('training_loss.png')
            plt.close()

if __name__ == '__main__':
    train()
