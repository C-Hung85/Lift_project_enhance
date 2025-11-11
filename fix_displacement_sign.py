#!/usr/bin/env python3
"""
修復 manual correction 產生 CSV 後距離欄位正負號的簡易工具。

用法：
    python fix_displacement_sign.py --csv lifts/result/mc1.csv

預設會在覆寫前建立一份 bak 備份，若希望輸出到另一個檔案，可加上 --output。
"""

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd


DISPLACEMENT_COLUMN_CANDIDATES = [
    "vertical_travel_distance (mm)",
    "vertical_travel_distance_mm",
    "vertical_travel_distance",
    "displacement",
    "displacement_mm",
    "位移",
    "位移_mm",
]


def find_displacement_column(df: pd.DataFrame) -> str:
    """嘗試依欄位名稱找出距離欄位。"""
    for col in DISPLACEMENT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"無法在 CSV 中找到距離欄位，請確認欄位名稱是否為以下候選之一：\n{DISPLACEMENT_COLUMN_CANDIDATES}"
    )


def fix_sign(df: pd.DataFrame, displacement_col: str, orientation_col: str) -> Tuple[int, int]:
    """依照 orientation 欄位調整距離欄位的正負號。

    Returns:
        (調整的列數, 總非零列數)
    """
    if orientation_col not in df.columns:
        raise ValueError(f"CSV 缺少 '{orientation_col}' 欄位，無法判定方向")

    displacement_series = df[displacement_col].copy()
    orientation_series = df[orientation_col]

    updated_values = []
    changed = 0
    non_zero_rows = 0

    for value, orientation in zip(displacement_series, orientation_series):
        if pd.isna(value) or value == 0 or pd.isna(orientation) or orientation == 0:
            updated_values.append(value)
            continue

        non_zero_rows += 1
        sign = 1 if orientation > 0 else -1
        new_value = abs(float(value)) * sign

        if abs(new_value - value) > 1e-9:
            changed += 1

        updated_values.append(new_value)

    df[displacement_col] = updated_values
    return changed, non_zero_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="根據 orientation 修正距離欄位正負號")
    parser.add_argument("--csv", required=True, help="要修正的 CSV 檔案路徑")
    parser.add_argument(
        "--output",
        help="輸出檔案路徑（預設覆寫原檔，並自動建立 .bak 備份）",
    )
    parser.add_argument(
        "--orientation-column",
        default="orientation",
        help="方向欄位名稱，預設為 orientation",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="以原檔為輸出時，不建立備份檔案",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到目標檔案：{csv_path}")

    output_path = Path(args.output).resolve() if args.output else csv_path

    df = pd.read_csv(csv_path)
    displacement_col = find_displacement_column(df)

    changed_rows, non_zero_rows = fix_sign(df, displacement_col, args.orientation_column)

    backup_path = None
    if output_path == csv_path and not args.no_backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        shutil.copy2(csv_path, backup_path)

    df.to_csv(output_path, index=False)

    print(f"✅ 修正完成：{changed_rows} 行更新，總共 {non_zero_rows} 行含非零距離。")
    if backup_path:
        print(f"   已建立備份檔案：{backup_path}")
    print(f"   輸出檔案：{output_path}")


if __name__ == "__main__":
    main()

