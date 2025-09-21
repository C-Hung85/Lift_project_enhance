"""
互動式影片旋轉角度設定工具
使用方式：python src/rotation_setup.py
"""
import os
import sys
import cv2
import random

# 設定路徑
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src/")

from config import Config, video_config
from rotation_utils import (
    rotate_frame, 
    get_video_frames_for_preview, 
    create_side_by_side_preview,
    normalize_filename,
    validate_video_exists
)


def setup_directories():
    """建立必要的目錄"""
    data_folder = Config['files']['data_folder']
    rotations_dir = os.path.join(data_folder, 'lifts', 'rotations')
    os.makedirs(rotations_dir, exist_ok=True)
    return data_folder, rotations_dir


def get_video_time_range(filename):
    """取得影片的時間範圍設定"""
    if filename in video_config:
        start_time = video_config[filename].get('start', 0)
        end_time = video_config[filename].get('end', None)
        return start_time, end_time
    return 0, None


def preview_rotation(video_path, filename, rotation_angle):
    """預覽旋轉效果"""
    try:
        # 取得影片資訊
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()
        
        # 取得時間範圍
        start_time, end_time = get_video_time_range(filename)
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else video_length
        
        print(f"從影片時間範圍 {start_time or 0}s - {end_time or int(video_length/fps)}s 中隨機選取預覽幀...")
        
        # 取得預覽幀
        frames = get_video_frames_for_preview(video_path, start_frame, end_frame, 2)
        
        if not frames:
            print("❌ 無法從影片中取得預覽幀")
            return False
        
        # 創建預覽圖片
        preview_image = create_side_by_side_preview(frames, rotation_angle)
        
        # 顯示預覽
        window_name = f"旋轉預覽 - {filename} ({rotation_angle}°)"
        cv2.imshow(window_name, preview_image)
        
        print(f"✅ 顯示預覽視窗: {window_name}")
        print("請查看預覽視窗，按任意鍵關閉預覽...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ 預覽失敗: {e}")
        return False


def get_user_choice():
    """取得使用者選擇"""
    print("\n預覽效果是否滿意？")
    print("  [y] 確認使用此角度")
    print("  [r] 重新輸入角度")
    print("  [c] 取消此影片設定")
    
    while True:
        choice = input("請選擇 (y/r/c): ").lower().strip()
        if choice in ['y', 'r', 'c']:
            return choice
        print("❌ 請輸入 y、r 或 c")


def get_rotation_angle():
    """取得旋轉角度輸入"""
    while True:
        try:
            angle_input = input("請輸入逆時針旋轉角度 (度): ").strip()
            if not angle_input:
                continue
            
            angle = float(angle_input)
            if -360 <= angle <= 360:
                return angle
            else:
                print("❌ 角度範圍應在 -360° 到 +360° 之間")
        except ValueError:
            print("❌ 請輸入有效的數字")


def save_rotation_preview(video_path, filename, rotation_angle, rotations_dir):
    """儲存旋轉預覽圖片"""
    try:
        # 取得預覽幀
        start_time, end_time = get_video_time_range(filename)
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vidcap.release()
        
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else video_length
        
        frames = get_video_frames_for_preview(video_path, start_frame, end_frame, 2)
        
        if frames:
            preview_image = create_side_by_side_preview(frames, rotation_angle)
            preview_filename = f"rotation_{filename}.png"
            preview_path = os.path.join(rotations_dir, preview_filename)
            
            cv2.imwrite(preview_path, preview_image)
            print(f"✅ 預覽圖片已儲存: {preview_path}")
            return True
    except Exception as e:
        print(f"⚠️  預覽圖片儲存失敗: {e}")
    
    return False


def update_rotation_config(rotation_settings):
    """更新旋轉配置檔案"""
    try:
        config_path = os.path.join("src", "rotation_config.py")
        
        # 準備配置內容
        config_lines = [
            "# 影片旋轉配置檔案",
            "# 格式：'影片檔名': 旋轉角度 (逆時針為正值)",
            "# 此檔案由 rotation_setup.py 自動產生和更新",
            "",
            "rotation_config = {"
        ]
        
        # 加入設定項目
        for filename, angle in rotation_settings.items():
            config_lines.append(f"    '{filename}': {angle},")
        
        config_lines.append("}")
        
        # 寫入檔案
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(config_lines))
        
        print(f"✅ 旋轉配置檔案已更新: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ 配置檔案更新失敗: {e}")
        return False


def main():
    """主要執行函數"""
    print("🔄 影片旋轉角度設定工具")
    print("=" * 50)
    
    # 設定目錄
    data_folder, rotations_dir = setup_directories()
    rotation_settings = {}
    
    while True:
        print(f"\n目前已設定 {len(rotation_settings)} 個影片的旋轉角度")
        
        # 取得影片檔名
        filename_input = input("\n請輸入要旋轉的影片檔名 (直接按 Enter 結束): ").strip()
        
        if not filename_input:
            break
        
        # 標準化檔名
        filename = normalize_filename(filename_input)
        
        # 驗證檔案存在
        exists, video_path = validate_video_exists(data_folder, filename)
        if not exists:
            print(f"❌ 找不到影片檔案: {video_path}")
            continue
        
        print(f"✅ 找到影片: {filename}")
        
        # 角度設定與預覽循環
        while True:
            # 取得旋轉角度
            rotation_angle = get_rotation_angle()
            
            # 顯示預覽
            print(f"\n🔍 產生旋轉預覽 ({rotation_angle}°)...")
            preview_success = preview_rotation(video_path, filename, rotation_angle)
            
            if not preview_success:
                print("❌ 預覽失敗，請重新輸入角度")
                continue
            
            # 取得使用者選擇
            choice = get_user_choice()
            
            if choice == 'y':
                # 確認使用此角度
                rotation_settings[filename] = rotation_angle
                print(f"✅ 已設定 {filename} 旋轉 {rotation_angle} 度")
                
                # 儲存預覽圖片
                save_rotation_preview(video_path, filename, rotation_angle, rotations_dir)
                break
                
            elif choice == 'c':
                # 取消此影片設定
                print(f"❌ 已取消 {filename} 的旋轉設定")
                break
                
            # choice == 'r' 時繼續循環重新輸入角度
    
    # 產生配置檔案
    if rotation_settings:
        print(f"\n📝 正在產生旋轉配置檔案...")
        print("設定摘要:")
        for filename, angle in rotation_settings.items():
            print(f"  • {filename}: {angle}°")
        
        if update_rotation_config(rotation_settings):
            print(f"\n🎉 設定完成！已產生 rotation_config.py")
            print(f"📁 預覽圖片存放於: {rotations_dir}")
        else:
            print(f"\n❌ 配置檔案產生失敗")
    else:
        print(f"\n⚠️  沒有設定任何旋轉角度")
    
    print("\n👋 感謝使用旋轉設定工具！")


if __name__ == "__main__":
    main()
