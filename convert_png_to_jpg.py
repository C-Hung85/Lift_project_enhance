#!/usr/bin/env python3
"""
PNG轉JPG畫質比較工具
將匯出的PNG檔案前10張轉換為JPG格式以便畫質比較
"""

import os
import cv2
import glob
from pathlib import Path

def convert_png_to_jpg(png_dir, num_files=10, jpg_quality=95):
    """
    將PNG檔案轉換為JPG格式

    Args:
        png_dir: PNG檔案目錄
        num_files: 轉換檔案數量
        jpg_quality: JPG壓縮品質 (1-100, 95為高品質)
    """

    # 確保目錄存在
    if not os.path.exists(png_dir):
        print(f"❌ 目錄不存在: {png_dir}")
        return

    # 取得所有PNG檔案並排序
    png_files = glob.glob(os.path.join(png_dir, "*.png"))
    png_files.sort()

    if not png_files:
        print(f"❌ 在 {png_dir} 中找不到PNG檔案")
        return

    print(f"📂 找到 {len(png_files)} 個PNG檔案")
    print(f"🔄 準備轉換前 {num_files} 個檔案...")

    # 創建JPG子目錄
    jpg_dir = os.path.join(png_dir, "jpg_comparison")
    os.makedirs(jpg_dir, exist_ok=True)

    converted_count = 0

    for i, png_file in enumerate(png_files[:num_files]):
        try:
            # 讀取PNG檔案
            img = cv2.imread(png_file)
            if img is None:
                print(f"⚠️  無法讀取: {os.path.basename(png_file)}")
                continue

            # 生成JPG檔名
            png_name = Path(png_file).stem
            jpg_name = f"{png_name}.jpg"
            jpg_path = os.path.join(jpg_dir, jpg_name)

            # 設定JPG壓縮參數
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]

            # 儲存為JPG
            success = cv2.imwrite(jpg_path, img, encode_param)

            if success:
                # 計算檔案大小
                png_size = os.path.getsize(png_file)
                jpg_size = os.path.getsize(jpg_path)
                compression_ratio = (png_size - jpg_size) / png_size * 100

                print(f"✅ {i+1:2d}. {png_name}")
                print(f"    PNG: {png_size/1024:.1f}KB -> JPG: {jpg_size/1024:.1f}KB (節省 {compression_ratio:.1f}%)")
                converted_count += 1
            else:
                print(f"❌ 轉換失敗: {png_name}")

        except Exception as e:
            print(f"❌ 處理 {os.path.basename(png_file)} 時發生錯誤: {e}")

    print(f"\n🎯 轉換完成！成功轉換 {converted_count}/{num_files} 個檔案")
    print(f"📁 JPG檔案位置: {jpg_dir}")

    # 顯示總體統計
    if converted_count > 0:
        total_png_size = sum(os.path.getsize(png_files[i]) for i in range(min(num_files, len(png_files))))
        total_jpg_size = sum(os.path.getsize(os.path.join(jpg_dir, f"{Path(png_files[i]).stem}.jpg"))
                           for i in range(converted_count)
                           if os.path.exists(os.path.join(jpg_dir, f"{Path(png_files[i]).stem}.jpg")))

        overall_compression = (total_png_size - total_jpg_size) / total_png_size * 100
        print(f"\n📊 整體統計:")
        print(f"   總PNG大小: {total_png_size/1024:.1f}KB")
        print(f"   總JPG大小: {total_jpg_size/1024:.1f}KB")
        print(f"   整體節省: {overall_compression:.1f}%")

def main():
    """主程式"""
    print("🖼️  PNG轉JPG畫質比較工具")
    print("=" * 40)

    # 設定參數
    png_directory = "lifts/exported_frames/1"
    num_files_to_convert = 10
    jpg_quality = 95  # 高品質JPG

    print(f"📂 來源目錄: {png_directory}")
    print(f"🔢 轉換數量: {num_files_to_convert}")
    print(f"🎨 JPG品質: {jpg_quality}")
    print()

    # 執行轉換
    convert_png_to_jpg(png_directory, num_files_to_convert, jpg_quality)

    print("\n💡 使用建議:")
    print("   1. 開啟 lifts/exported_frames/1/jpg_comparison/ 目錄")
    print("   2. 比較同名的PNG和JPG檔案")
    print("   3. 注意畫質差異和檔案大小")

if __name__ == "__main__":
    main()