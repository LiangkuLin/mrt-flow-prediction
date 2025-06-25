import os
import pandas as pd

# 設定你的資料夾路徑
folder_path = '資料集'  # 例如：'./csv_files'

# 找出資料夾內所有的 CSV 檔案
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

combined_df = pd.DataFrame()

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path, encoding='utf-8')  # 嘗試用 utf-8 讀
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='big5')  # 如果失敗，改用 big5 讀
    combined_df = pd.concat([combined_df, df], ignore_index=True)

output_path = os.path.join(folder_path, 'combined_output.csv')
combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"成功合併 {len(csv_files)} 個檔案，輸出檔案位於：{output_path}")