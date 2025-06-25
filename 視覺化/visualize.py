import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader

# === 設定 ===
SEQ_LEN = 30
CHECKPOINT_PATH = './checkpoints_v2/checkpoint_epoch_1000.pt'
ERROR_THRESHOLD = 700000  # 可自行調整誤差門檻

# === 資料集類別 ===
class MetroDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.seq_len = seq_len
        self.features = data.drop(columns=['總運量']).values
        self.targets = data['總運量'].values

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === 模型定義 ===
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze()

# === 資料讀取與標準化 ===
def load_data():
    df = pd.read_csv('features_ready.csv')
    df = df.dropna()
    df['是否為節日'] = df['是否為節日'].map({'是': 1, '否': 0})
    df['日期'] = pd.to_datetime(df['日期'])

    df = df[['日期', '總運量', '星期幾', '是否週末', '月份', '是否寒暑假',
             '是否連假前一日', '是否連假最後日', '是否為節日',
             '滾動平均_7日', '滾動平均_14日', '去年同日運量']]

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    mean = df.iloc[:split_idx].drop(columns=['日期']).mean()
    std = df.iloc[:split_idx].drop(columns=['日期']).std()

    test_df_scaled = test_df.copy()
    test_df_scaled.iloc[:, 1:] = (test_df_scaled.iloc[:, 1:] - mean) / std

    return test_df, test_df_scaled, mean, std

# === 評估與繪圖 ===
def evaluate_and_plot():
    test_df, test_df_scaled, mean, std = load_data()
    test_dataset = MetroDataset(test_df_scaled.drop(columns=['日期']), seq_len=SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = RNNModel(input_size=10)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).item()
            preds.append(pred)
            trues.append(y.item())

    preds = np.array(preds) * std['總運量'] + mean['總運量']
    trues = np.array(trues) * std['總運量'] + mean['總運量']
    dates = test_df['日期'].values[SEQ_LEN:]
    errors = np.abs(preds - trues)

    # === 繪圖 ===
    plt.figure(figsize=(14, 6))
    plt.plot(dates, trues, label='True')
    plt.plot(dates, preds, label='Predicted')

    # 所有超過誤差門檻的點都標出
    for i in range(len(errors)):
        if errors[i] > ERROR_THRESHOLD:
            date = dates[i]
            plt.scatter(date, trues[i], color='red', s=30)
            plt.text(date, trues[i], f'{str(date)[:10]}\nErr: {int(errors[i]):,}',
                     fontsize=7, color='red', ha='right', va='bottom')

    plt.title('Metro Usage Prediction')
    plt.xlabel('Date')
    plt.ylabel('Total Volume')
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('prediction_vs_true_by_date_all_outliers.png')
    plt.show()

if __name__ == '__main__':
    evaluate_and_plot()