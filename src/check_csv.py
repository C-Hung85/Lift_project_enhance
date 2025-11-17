#!/usr/bin/env python3
"""檢查 CSV 檔案結構"""
import pandas as pd

csv_path = 'lifts/result/1.csv'
df = pd.read_csv(csv_path)

print("CSV columns:", df.columns.tolist())
print("CSV shape:", df.shape)
print("\nframe_path value counts:")
print(df['frame_path'].value_counts(dropna=False).head(30))
print(f"\n總 NaN: {df['frame_path'].isna().sum()}")
print(f"總非 NaN: {df['frame_path'].notna().sum()}")
print("\n前30個 frame_path 值:")
print(df['frame_path'].unique()[:30])


