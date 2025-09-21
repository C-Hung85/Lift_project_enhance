"""
比例尺快取管理工具
使用方式：python src/scale_setup.py
"""
import os
import sys

# 設定路徑
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.append("src/")

from config import Config
from scale_cache_utils import (
    load_scale_cache, 
    save_scale_cache, 
    print_cache_status,
    generate_scale_images_hash
)


def show_cache_info():
    """顯示快取資訊"""
    scale_cache, cache_info = load_scale_cache()
    
    print("📊 比例尺快取資訊:")
    print("=" * 60)
    
    if cache_info:
        print(f"📅 最後更新: {cache_info.get('last_updated', '未知')}")
        print(f"📁 目錄雜湊: {cache_info.get('directory_hash', '未知')[:16]}...")
        print(f"🆕 上次處理: {cache_info.get('newly_processed', 0)} 個影片")
    else:
        print("⚠️  沒有快取資訊")
    
    print_cache_status(scale_cache)
    

def clear_cache():
    """清除比例尺快取"""
    confirm = input("⚠️  確定要清除所有比例尺快取嗎？ [y/N]: ").lower().strip()
    
    if confirm in ['y', 'yes']:
        # 儲存空的快取
        cache_info = {
            'last_updated': '',
            'total_videos': 0,
            'directory_hash': '',
            'action': 'cache_cleared'
        }
        
        if save_scale_cache({}, cache_info):
            print("✅ 比例尺快取已清除")
        else:
            print("❌ 清除快取失敗")
    else:
        print("❌ 已取消清除操作")


def rebuild_cache():
    """重建比例尺快取"""
    data_folder = Config['files']['data_folder']
    scale_images_dir = os.path.join(data_folder, 'lifts', 'scale_images')
    
    print("🔄 重建比例尺快取...")
    print("⚠️  這將重新計算所有影片的比例尺")
    
    confirm = input("確定要繼續嗎？ [y/N]: ").lower().strip()
    
    if confirm in ['y', 'yes']:
        # 清除現有快取
        cache_info = {
            'last_updated': '',
            'total_videos': 0,
            'directory_hash': generate_scale_images_hash(scale_images_dir),
            'action': 'cache_rebuild'
        }
        
        if save_scale_cache({}, cache_info):
            print("✅ 快取已清除，請執行主程式重新計算比例尺")
            print("📌 執行指令: python src/lift_travel_detection.py")
        else:
            print("❌ 重建快取失敗")
    else:
        print("❌ 已取消重建操作")


def show_specific_video(video_name):
    """顯示特定影片的比例尺資訊"""
    scale_cache, cache_info = load_scale_cache()
    
    # 標準化檔名
    if not video_name.endswith('.mp4'):
        video_name += '.mp4'
    
    if video_name in scale_cache:
        scale_value = scale_cache[video_name]
        print(f"📹 {video_name}: {scale_value:.6f} 像素")
    else:
        print(f"❌ 找不到影片 {video_name} 的比例尺資料")
        
        # 列出可用的影片
        if scale_cache:
            print("\n📋 可用的影片:")
            for video in sorted(scale_cache.keys()):
                print(f"  📹 {video}")


def remove_video_cache(video_name):
    """移除特定影片的快取"""
    scale_cache, cache_info = load_scale_cache()
    
    # 標準化檔名
    if not video_name.endswith('.mp4'):
        video_name += '.mp4'
    
    if video_name in scale_cache:
        print(f"🗑️  移除影片快取: {video_name} (原值: {scale_cache[video_name]:.6f})")
        
        confirm = input("確定要移除嗎？ [y/N]: ").lower().strip()
        if confirm in ['y', 'yes']:
            del scale_cache[video_name]
            
            # 更新快取資訊
            if cache_info:
                cache_info['last_updated'] = ''
                cache_info['total_videos'] = len(scale_cache)
                cache_info['action'] = f'removed_{video_name}'
            
            if save_scale_cache(scale_cache, cache_info):
                print(f"✅ 已移除 {video_name} 的快取")
            else:
                print("❌ 移除快取失敗")
        else:
            print("❌ 已取消移除操作")
    else:
        print(f"❌ 找不到影片 {video_name} 的快取資料")


def main():
    """主要執行函數"""
    print("📏 比例尺快取管理工具")
    print("=" * 60)
    
    while True:
        print("\n選擇操作:")
        print("  [1] 顯示快取資訊")
        print("  [2] 清除所有快取")
        print("  [3] 重建快取")
        print("  [4] 查看特定影片")
        print("  [5] 移除特定影片快取")
        print("  [q] 退出")
        
        choice = input("\n請選擇 (1-5/q): ").strip().lower()
        
        if choice == '1':
            show_cache_info()
            
        elif choice == '2':
            clear_cache()
            
        elif choice == '3':
            rebuild_cache()
            
        elif choice == '4':
            video_name = input("請輸入影片檔名 (可省略 .mp4): ").strip()
            if video_name:
                show_specific_video(video_name)
            else:
                print("❌ 請輸入有效的影片檔名")
                
        elif choice == '5':
            video_name = input("請輸入要移除的影片檔名 (可省略 .mp4): ").strip()
            if video_name:
                remove_video_cache(video_name)
            else:
                print("❌ 請輸入有效的影片檔名")
                
        elif choice in ['q', 'quit', 'exit']:
            print("👋 感謝使用比例尺快取管理工具！")
            break
            
        else:
            print("❌ 無效的選擇，請重新輸入")


if __name__ == "__main__":
    main()
