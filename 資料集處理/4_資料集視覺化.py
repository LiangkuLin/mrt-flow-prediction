import pandas as pd
import matplotlib.pyplot as plt
import os

# 建立資料夾
output_dir = "資料集處理/資料集視覺化"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示成亂碼


df = pd.read_csv("資料集處理/資料集/features_ready.csv")


category_cols = ['是否為節日', '是否週末', '是否寒暑假', '是否連假前一日', '是否連假最後日']

for col in category_cols:
    counts = df[col].value_counts().sort_index()
    plt.figure()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f'{col} 分布圖')
    plt.xlabel(col)
    plt.ylabel('數量')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, f'{col}_bar_chart.png'), bbox_inches='tight', dpi=300)
    # plt.show()


numeric_cols = ['總運量', '滾動平均_7日', '滾動平均_14日']

for col in numeric_cols:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f'{col} 直方圖')
    plt.xlabel(col)
    plt.ylabel('頻率')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{col}_bar_chart.png'), bbox_inches='tight', dpi=300)
    # plt.show()

    plt.figure()
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f'{col} 盒鬍圖')
    plt.xlabel(col)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{col}_box_plot.png'), bbox_inches='tight', dpi=300)
    # plt.show()


# 日期欄轉換格式並排序
df['日期'] = pd.to_datetime(df['日期'])
df = df.sort_values('日期')

# 畫出時間序列圖
plt.figure(figsize=(14, 5))
plt.plot(df['日期'], df['總運量'], label='總運量')
plt.plot(df['日期'], df['滾動平均_7日'], label='7日平均')
plt.plot(df['日期'], df['滾動平均_14日'], label='14日平均')
plt.title('日期 vs 運量與滾動平均')
plt.xlabel('日期')
plt.ylabel('運量')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'date_vs_volume_and_moving_average.png'), bbox_inches='tight', dpi=300)
# plt.show()
