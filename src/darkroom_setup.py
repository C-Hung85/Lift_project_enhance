"""
互動式暗房時間區間設定工具
使用方式：python src/darkroom_setup.py
"""
import os
import sys
import re

# 設定路徑
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src/")

from config import Config
from darkroom_utils import (
    parse_time_string, 
    convert_intervals_to_seconds,
    validate_darkroom_config,
    print_darkroom_summary
)

# 嘗試載入現有的暗房配置
try:
    from darkroom_intervals import darkroom_intervals
except ImportError:
    # 如果 darkroom_intervals.py 不存在，使用空字典
    darkroom_intervals = {}


def normalize_filename(filename):
    """標準化檔名，自動加入 .mp4 副檔名"""
    filename = filename.strip()
    if not filename:
        return filename
    
    if not filename.endswith('.mp4'):
        filename += '.mp4'
    
    return filename


def validate_video_exists(data_folder, filename):
    """驗證影片檔案是否存在"""
    video_path = os.path.join(data_folder, 'lifts', 'data', filename)
    exists = os.path.exists(video_path)
    return exists, video_path


def parse_time_input(time_input):
    """
    解析時間輸入，支援多種格式
    支援格式：MM:SS, M:SS, HH:MM:SS
    """
    time_input = time_input.strip()
    
    # 移除可能的空格
    time_input = re.sub(r'\s+', '', time_input)
    
    try:
        return parse_time_string(time_input)
    except ValueError:
        return None


def get_time_interval():
    """取得一個時間區間的輸入"""
    while True:
        print("\n📅 輸入時間區間:")
        
        # 取得開始時間
        while True:
            start_input = input("  開始時間 (MM:SS): ").strip()
            if not start_input:
                return None  # 使用者想要結束
            
            start_seconds = parse_time_input(start_input)
            if start_seconds is not None:
                break
            print("  ❌ 時間格式錯誤，請使用 MM:SS 格式")
        
        # 取得結束時間
        while True:
            end_input = input("  結束時間 (MM:SS): ").strip()
            if not end_input:
                print("  ❌ 請輸入結束時間")
                continue
            
            end_seconds = parse_time_input(end_input)
            if end_seconds is not None:
                if end_seconds > start_seconds:
                    break
                else:
                    print("  ❌ 結束時間必須大於開始時間")
            else:
                print("  ❌ 時間格式錯誤，請使用 MM:SS 格式")
        
        # 確認時間區間
        duration = end_seconds - start_seconds
        duration_minutes = duration // 60
        duration_seconds = duration % 60
        
        print(f"  ✅ 時間區間: {start_input} ~ {end_input} (持續 {duration_minutes:02d}:{duration_seconds:02d})")
        
        confirm = input("  確認此時間區間？ [y/n]: ").lower().strip()
        if confirm in ['y', 'yes', '']:
            return (start_input, end_input)
        print("  🔄 重新輸入時間區間...")


def get_intervals_for_video():
    """取得一個影片的所有暗房時間區間"""
    intervals = []
    
    print("\n💡 提示：可設定多個時間區間，開始時間留空結束輸入")
    
    while True:
        interval = get_time_interval()
        if interval is None:
            break
        
        intervals.append(interval)
        print(f"  ✅ 已加入區間 {len(intervals)}: {interval[0]} ~ {interval[1]}")
        
        if len(intervals) >= 5:  # 限制最多5個區間
            print("  ⚠️  已達到最大區間數量限制 (5個)")
            break
    
    return intervals


def show_current_settings(darkroom_settings):
    """顯示目前的設定"""
    if not darkroom_settings:
        print("\n📋 目前沒有任何暗房區間設定")
        return
    
    print(f"\n📋 目前已設定 {len(darkroom_settings)} 個影片的暗房區間:")
    print("-" * 50)
    
    for filename, intervals in darkroom_settings.items():
        print(f"🎬 {filename}:")
        if not intervals:
            print("  (無區間)")
        else:
            for i, (start, end) in enumerate(intervals, 1):
                print(f"  🌙 區間 {i}: {start} ~ {end}")
        print()


def update_darkroom_config(darkroom_settings):
    """更新暗房配置檔案"""
    try:
        config_path = os.path.join("src", "darkroom_intervals.py")
        
        # 準備配置內容
        config_lines = [
            "# 暗房區間配置檔案",
            "# 格式：'影片檔名': [('開始時間', '結束時間'), ...]",
            "# 時間格式：'MM:SS' (分:秒)",
            "# 在這些時間區間內，所有運動偵測結果都會被忽略（類似 camera pan）",
            "",
            "darkroom_intervals = {"
        ]
        
        # 加入設定項目
        for filename, intervals in darkroom_settings.items():
            if intervals:  # 只有非空的區間才寫入
                intervals_str = ", ".join([f"('{start}', '{end}')" for start, end in intervals])
                config_lines.append(f"    '{filename}': [{intervals_str}],")
        
        config_lines.extend([
            "",
            "    # 範例格式說明：",
            "    # '影片檔名.mp4': [('開始時間', '結束時間'), ('開始時間2', '結束時間2'), ...]",
            "    # 支援多個時間區間",
            "}"
        ])
        
        # 寫入檔案
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(config_lines))
        
        print(f"✅ 暗房配置檔案已更新: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ 配置檔案更新失敗: {e}")
        return False


def main():
    """主要執行函數"""
    print("🌙 暗房時間區間設定工具")
    print("=" * 50)
    print("💡 用於設定影片中需要忽略運動偵測的暗房時間區間")
    
    # 設定目錄
    data_folder = Config['files']['data_folder']
    
    # 載入現有的暗房設定（深拷貝以避免修改原始數據）
    darkroom_settings = {}
    for filename, intervals in darkroom_intervals.items():
        darkroom_settings[filename] = list(intervals)  # 複製列表
    
    if darkroom_settings:
        print(f"\n📂 已載入現有設定，共 {len(darkroom_settings)} 個影片")
    
    while True:
        show_current_settings(darkroom_settings)
        
        # 取得操作選項
        print("\n操作選項:")
        print("  1. 新增/修改影片的暗房區間")
        print("  2. 刪除影片的暗房設定")
        print("  3. 完成設定")
        
        choice = input("請選擇操作 [1/2/3]: ").strip()
        
        if choice == '3' or choice == '':
            break
        elif choice == '2':
            if not darkroom_settings:
                print("❌ 沒有任何設定可以刪除")
                continue
            
            print("\n現有設定:")
            filenames = list(darkroom_settings.keys())
            for i, filename in enumerate(filenames, 1):
                print(f"  {i}. {filename}")
            
            try:
                del_choice = input("請輸入要刪除的影片編號 (或按 Enter 取消): ").strip()
                if del_choice:
                    del_index = int(del_choice) - 1
                    if 0 <= del_index < len(filenames):
                        del_filename = filenames[del_index]
                        del darkroom_settings[del_filename]
                        print(f"✅ 已刪除 {del_filename} 的暗房設定")
                    else:
                        print("❌ 編號無效")
            except ValueError:
                print("❌ 請輸入有效的編號")
            continue
        elif choice != '1':
            print("❌ 無效的選擇，請重新選擇")
            continue
        
        # 取得影片檔名
        filename_input = input("\n請輸入要設定暗房區間的影片檔名 (直接按 Enter 返回選單): ").strip()
        
        if not filename_input:
            continue
        
        # 標準化檔名
        filename = normalize_filename(filename_input)
        
        # 驗證檔案存在
        exists, video_path = validate_video_exists(data_folder, filename)
        if not exists:
            print(f"❌ 找不到影片檔案: {video_path}")
            continue
        
        print(f"✅ 找到影片: {filename}")
        
        # 如果已經有設定，詢問是否要修改
        if filename in darkroom_settings:
            print(f"⚠️  此影片已有暗房區間設定：")
            for i, (start, end) in enumerate(darkroom_settings[filename], 1):
                print(f"    區間 {i}: {start} ~ {end}")
            
            action = input("選擇動作 [a]新增區間 [r]重新設定 [s]跳過: ").lower().strip()
            if action == 's':
                continue
            elif action == 'r':
                darkroom_settings[filename] = []
            # action == 'a' 時繼續現有設定
        else:
            darkroom_settings[filename] = []
        
        # 取得暗房時間區間
        new_intervals = get_intervals_for_video()
        
        if new_intervals:
            darkroom_settings[filename].extend(new_intervals)
            print(f"✅ 已為 {filename} 設定 {len(new_intervals)} 個暗房區間")
        else:
            if filename in darkroom_settings and not darkroom_settings[filename]:
                del darkroom_settings[filename]  # 移除空設定
            print(f"❌ 沒有為 {filename} 設定任何暗房區間")
    
    # 最終確認和產生配置檔案
    if darkroom_settings:
        # 比較原始設定和新設定
        original_count = len(darkroom_intervals)
        current_count = len(darkroom_settings)
        
        if original_count > 0:
            print(f"\n📝 正在更新暗房配置檔案...")
            print(f"原有設定: {original_count} 個影片")
            print(f"更新後: {current_count} 個影片")
        else:
            print(f"\n📝 正在建立暗房配置檔案...")
            print(f"新設定: {current_count} 個影片")
        
        # 驗證配置
        is_valid, errors = validate_darkroom_config(darkroom_settings)
        if not is_valid:
            print("❌ 配置驗證失敗:")
            for error in errors:
                print(f"  • {error}")
            return
        
        print_darkroom_summary(darkroom_settings)
        
        action_word = "更新" if original_count > 0 else "建立"
        final_confirm = input(f"\n確認{action_word}配置檔案？ [y/n]: ").lower().strip()
        if final_confirm in ['y', 'yes', '']:
            if update_darkroom_config(darkroom_settings):
                print(f"\n🎉 設定完成！已{action_word} darkroom_intervals.py")
                print("💡 執行主程式時會自動忽略這些時間區間的運動偵測")
            else:
                print(f"\n❌ 配置檔案{action_word}失敗")
        else:
            print(f"\n❌ 已取消配置檔案{action_word}")
    else:
        if len(darkroom_intervals) > 0:
            print(f"\n⚠️  所有暗房區間設定已被刪除")
            clear_confirm = input("確認清空配置檔案？ [y/n]: ").lower().strip()
            if clear_confirm in ['y', 'yes', '']:
                if update_darkroom_config({}):
                    print(f"\n✅ 配置檔案已清空")
                else:
                    print(f"\n❌ 配置檔案清空失敗")
            else:
                print(f"\n❌ 已取消清空操作")
        else:
            print(f"\n⚠️  沒有設定任何暗房區間")
    
    print("\n👋 感謝使用暗房區間設定工具！")


if __name__ == "__main__":
    main()
