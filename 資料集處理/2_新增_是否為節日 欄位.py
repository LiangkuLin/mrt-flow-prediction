"""
https://data.taipei/dataset/detail?id=c30ca421-d935-4faa-b523-9c175c8de738
"""

import pandas as pd

# === Step 1: 讀取合併後的主資料集 ===
df = pd.read_csv('資料集/combined_output.csv')

# 將年月日轉為 datetime 格式
df['日期'] = pd.to_datetime(df['西元年'].astype(str) + '-' +
                            df['月'].astype(str).str.zfill(2) + '-' +
                            df['日'].astype(str).str.zfill(2))


# === Step 2: 讀取節日檔案（你下載的 holiday.csv）===
holiday_df = pd.read_csv('資料集/政府行政機關辦公日曆表102年至114年.csv', dtype={'Date': str})

# 將 Date 欄轉為 datetime
holiday_df['日期'] = pd.to_datetime(holiday_df['Date'], format='%Y%m%d')

# 只保留日期和是否為節日欄位
holiday_df = holiday_df[['日期', 'isHoliday']]

# === Step 3: 合併節日資料 ===
df = df.merge(holiday_df, on='日期', how='left')

# 將空值填為 '否'
df['是否為節日'] = df['isHoliday'].fillna('否')

# 移除原始 isHoliday 欄（可選）
df = df.drop(columns=['isHoliday'])

# === Step 4: 儲存最終結果 ===
df.to_csv('資料集/combined_with_holiday.csv', index=False, encoding='utf-8-sig')

print("✅ 已成功加入節日資訊並儲存為 '資料集/combined_with_holiday.csv'")
