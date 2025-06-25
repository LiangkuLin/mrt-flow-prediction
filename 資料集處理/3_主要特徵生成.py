import pandas as pd

# 讀資料
df = pd.read_csv('資料集/combined_with_holiday.csv')

# 建立 datetime 欄位
df['日期'] = pd.to_datetime(df['西元年'].astype(str) + '-' +
                            df['月'].astype(str).str.zfill(2) + '-' +
                            df['日'].astype(str).str.zfill(2))

# 依日期排序
df = df.sort_values('日期')

# 基本時間特徵
df['星期幾'] = df['日期'].dt.weekday  # 0=週一
df['是否週末'] = df['星期幾'].isin([5, 6]).astype(int)
df['月份'] = df['日期'].dt.month

# 寒暑假（簡化版本）
df['是否寒暑假'] = df['月份'].isin([1, 2, 7, 8]).astype(int)

# 滾動平均
df['滾動平均_7日'] = df['總運量'].rolling(window=7).mean()
df['滾動平均_14日'] = df['總運量'].rolling(window=14).mean()

# 去年同日運量
df['去年同日'] = df['日期'] - pd.DateOffset(years=1)
df = df.merge(df[['日期', '總運量']].rename(columns={'日期': '去年同日', '總運量': '去年同日運量'}),
              on='去年同日', how='left')

# 是否連假前一日
df['是否連假前一日'] = df['是否為節日'].shift(-1).eq('是').astype(int)

# 是否連假最後一日
df['是否連假最後日'] = ((df['是否為節日'] == '否') & (df['是否為節日'].shift(1) == '是')).astype(int)

# 篩掉一開始無法計算滾動平均的列
df = df.dropna(subset=['滾動平均_7日', '滾動平均_14日'])

# 儲存特徵表
df.to_csv('資料集/features_ready.csv', index=False, encoding='utf-8-sig')

print("✅ 特徵表已成功建立，儲存為：資料集/features_ready.csv")
