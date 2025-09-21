"""
比例尺快取預建立工具
專門用於預先計算和建立比例尺快取，不執行影片分析
使用方式：python src/build_scale_cache.py
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# 設定路徑
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src/")

from config import Config
try:
    from rotation_config import rotation_config
except ImportError:
    rotation_config = {}

from scale_cache_utils import (
    save_scale_cache, 
    generate_scale_images_hash,
    print_cache_status
)


def main():
    """預建立比例尺快取"""
    print("📏 比例尺快取預建立工具")
    print("=" * 50)
    print("🔄 正在計算所有影片的比例尺...")
    
    DATA_FOLDER = Config['files']['data_folder']
    scale_images_dir = os.path.join(DATA_FOLDER, 'lifts', 'scale_images')
    
    # 處理比例尺圖片
    new_scale_data = {}
    processed_count = 0
    error_count = 0
    rotation_warning_count = 0
    
    for root, folder, files in os.walk(scale_images_dir):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            video_name = "-".join(file.split(sep="-")[:-1]) + ".mp4"
            
            print(f"🔄 處理: {file} → {video_name}")
            
            try:
                image = cv2.imread(os.path.join(root, file))
                if image is None:
                    print(f"  ❌ 無法讀取圖片")
                    error_count += 1
                    continue
                
                # 從原始圖片中尋找紅色標記點
                filtered_array = (image[..., 0] < 10) * (image[..., 1] < 10) * (image[..., 2] > 250)
                points = np.where(filtered_array)
                
                # 檢查是否找到足夠的紅色標記點
                if len(points[0]) < 2:
                    print(f"  ❌ 找不到足夠的紅色標記點 (找到 {len(points[0])} 個)")
                    error_count += 1
                    continue
                
                # 取得兩個紅點的座標 (y, x)
                point1_original = (points[0][0], points[1][0])  # (y1, x1)
                point2_original = (points[0][1], points[1][1])  # (y2, x2)
                
                # 計算原始歐氏距離（作為驗算基準）
                original_euclidean_distance = np.sqrt(
                    (point1_original[0] - point2_original[0])**2 + 
                    (point1_original[1] - point2_original[1])**2
                )
                
                # 複製座標用於旋轉計算
                point1 = point1_original
                point2 = point2_original
                
                # 如果影片需要旋轉，對紅點座標進行相應的旋轉變換
                if video_name in rotation_config:
                    rotation_angle = rotation_config[video_name]
                    print(f"  🔄 套用旋轉: {rotation_angle}°")
                    
                    # 取得圖片中心點
                    h, w = image.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # 將角度轉換為弧度
                    angle_rad = np.radians(rotation_angle)
                    cos_angle = np.cos(angle_rad)
                    sin_angle = np.sin(angle_rad)
                    
                    # 對兩個點進行旋轉變換
                    def rotate_point(y, x, center_y, center_x, cos_a, sin_a):
                        # 將座標移至原點
                        rel_x = x - center_x
                        rel_y = y - center_y
                        # 進行旋轉（注意座標系統：影像 y 軸向下）
                        new_x = rel_x * cos_a + rel_y * sin_a
                        new_y = -rel_x * sin_a + rel_y * cos_a
                        # 移回原位置
                        return new_y + center_y, new_x + center_x
                    
                    point1 = rotate_point(point1[0], point1[1], center_y, center_x, cos_angle, sin_angle)
                    point2 = rotate_point(point2[0], point2[1], center_y, center_x, cos_angle, sin_angle)
                
                # 計算垂直方向距離（y 軸距離）
                distance = abs(point1[0] - point2[0])
                
                # 旋轉驗算：檢查垂直距離與原始歐氏距離的差異
                if video_name in rotation_config:
                    # 計算差異百分比
                    difference_ratio = abs(distance - original_euclidean_distance) / original_euclidean_distance
                    difference_percent = difference_ratio * 100
                    
                    print(f"    📏 原始歐氏距離: {original_euclidean_distance:.2f} 像素")
                    print(f"    📐 旋轉後垂直距離: {distance:.2f} 像素")
                    print(f"    📊 差異百分比: {difference_percent:.1f}%")
                    
                    # 如果差異超過 10% 發出警告
                    if difference_percent > 10.0:
                        print(f"    ⚠️  旋轉驗算警告: 垂直距離與原始距離差異過大 ({difference_percent:.1f}%)")
                        print(f"        可能原因: 旋轉計算錯誤或紅點位置不當")
                        print(f"        建議檢查: {file} 的紅點標記位置")
                        rotation_warning_count += 1
                    else:
                        print(f"    ✅ 旋轉驗算通過")
                else:
                    # 沒有旋轉時，使用原始垂直距離
                    original_vertical_distance = abs(point1_original[0] - point2_original[0])
                    distance = original_vertical_distance
                    print(f"    📏 原始垂直距離: {distance:.2f} 像素")
                    print(f"    📐 原始歐氏距離: {original_euclidean_distance:.2f} 像素")
                
                print(f"  ✅ 垂直距離: {distance:.2f} 像素")
                
                if video_name in new_scale_data:
                    new_scale_data[video_name].append(distance)
                else:
                    new_scale_data[video_name] = [distance]
                
                processed_count += 1
                
            except Exception as e:
                print(f"  ❌ 處理錯誤: {e}")
                error_count += 1
                continue
    
    # 計算平均值
    scale_cache = {}
    for video_name, distances in new_scale_data.items():
        scale_cache[video_name] = np.mean(distances)
        print(f"📊 {video_name}: 平均比例尺 {scale_cache[video_name]:.4f} 像素 (來自 {len(distances)} 張圖片)")
    
    # 儲存快取
    if scale_cache:
        cache_info = {
            'last_updated': datetime.now().isoformat(),
            'total_videos': len(scale_cache),
            'directory_hash': generate_scale_images_hash(scale_images_dir),
            'processed_images': processed_count,
            'error_count': error_count,
            'rotation_warnings': rotation_warning_count,
            'build_method': 'pre_build'
        }
        
        if save_scale_cache(scale_cache, cache_info):
            print(f"\n🎉 比例尺快取建立完成！")
            print(f"✅ 成功處理: {len(scale_cache)} 個影片")
            print(f"📊 處理圖片: {processed_count} 張")
            if error_count > 0:
                print(f"❌ 錯誤圖片: {error_count} 張")
            if rotation_warning_count > 0:
                print(f"⚠️  旋轉驗算警告: {rotation_warning_count} 張")
                print(f"💡 建議檢查有警告的圖片，確認紅點位置正確")
            else:
                print(f"✅ 所有旋轉驗算都通過")
            
            print_cache_status(scale_cache)
            
            print(f"\n💡 現在執行主程式會直接使用快取，大幅提升速度！")
            print(f"📌 執行指令: python src/lift_travel_detection.py")
        else:
            print(f"\n❌ 快取儲存失敗")
    else:
        print(f"\n❌ 沒有成功處理任何比例尺圖片")
        print(f"⚠️  錯誤數量: {error_count}")
        print(f"💡 請檢查比例尺圖片是否包含紅色標記點")


if __name__ == "__main__":
    main()
