"""
比例尺快取管理工具函數模組
"""
import os
import hashlib
import numpy as np
from datetime import datetime


def generate_scale_images_hash(scale_images_dir):
    """
    生成比例尺圖片目錄的雜湊值，用於檢查是否有變更
    
    Args:
        scale_images_dir: 比例尺圖片目錄路徑
    
    Returns:
        hash_value: 目錄內容的 MD5 雜湊值
    """
    hash_md5 = hashlib.md5()
    
    try:
        for root, dirs, files in os.walk(scale_images_dir):
            for file in sorted(files):  # 排序確保一致性
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    # 加入檔名和修改時間
                    hash_md5.update(file.encode('utf-8'))
                    hash_md5.update(str(os.path.getmtime(file_path)).encode('utf-8'))
        
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"⚠️  生成雜湊值時發生錯誤: {e}")
        return None


def load_scale_cache():
    """
    載入比例尺快取配置
    
    Returns:
        scale_config: 比例尺配置字典
        cache_info: 快取資訊字典
    """
    try:
        # 動態載入配置
        import importlib.util
        spec = importlib.util.spec_from_file_location("scale_config", "src/scale_config.py")
        scale_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scale_config_module)
        
        scale_config = getattr(scale_config_module, 'scale_config', {})
        cache_version = getattr(scale_config_module, 'CACHE_VERSION', '1.0')
        
        # 嘗試載入快取資訊
        cache_info = {}
        if hasattr(scale_config_module, 'CACHE_INFO'):
            cache_info = scale_config_module.CACHE_INFO
        
        return scale_config, cache_info
        
    except Exception as e:
        print(f"⚠️  載入比例尺快取時發生錯誤: {e}")
        return {}, {}


def save_scale_cache(scale_config, cache_info=None):
    """
    儲存比例尺快取配置
    
    Args:
        scale_config: 比例尺配置字典
        cache_info: 快取資訊字典
    """
    try:
        config_path = os.path.join("src", "scale_config.py")
        
        # 準備快取資訊
        if cache_info is None:
            cache_info = {
                'last_updated': datetime.now().isoformat(),
                'total_videos': len(scale_config)
            }
        
        # 準備配置內容
        config_lines = [
            "# 比例尺快取配置檔案",
            "# 格式：'影片檔名': 平均距離像素值",
            "# 此檔案由程式自動產生和更新，避免重複計算比例尺",
            "",
            "scale_config = {"
        ]
        
        # 加入配置項目（按檔名排序）
        for filename in sorted(scale_config.keys()):
            scale_value = scale_config[filename]
            config_lines.append(f"    '{filename}': {scale_value:.6f},")
        
        config_lines.extend([
            "}",
            "",
            "# 快取版本號，用於檢查快取是否需要更新",
            'CACHE_VERSION = "1.0"',
            "",
            "# 快取資訊",
            f"CACHE_INFO = {repr(cache_info)}"
        ])
        
        # 寫入檔案
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(config_lines))
        
        print(f"✅ 比例尺快取已更新: {config_path}")
        print(f"📊 快取統計: {len(scale_config)} 個影片")
        return True
        
    except Exception as e:
        print(f"❌ 比例尺快取儲存失敗: {e}")
        return False


def is_cache_valid(scale_images_dir, cache_info):
    """
    檢查比例尺快取是否仍然有效
    
    Args:
        scale_images_dir: 比例尺圖片目錄
        cache_info: 快取資訊
    
    Returns:
        is_valid: 快取是否有效
    """
    if not cache_info:
        return False
    
    # 檢查目錄雜湊值
    current_hash = generate_scale_images_hash(scale_images_dir)
    cached_hash = cache_info.get('directory_hash')
    
    if current_hash != cached_hash:
        print("🔄 偵測到比例尺圖片變更，需要重新計算")
        return False
    
    return True


def get_missing_videos(scale_config, video_files):
    """
    取得需要計算比例尺的影片列表
    
    Args:
        scale_config: 目前的比例尺配置
        video_files: 所有影片檔案列表
    
    Returns:
        missing_videos: 缺少比例尺的影片列表
    """
    missing_videos = []
    
    for video_file in video_files:
        if video_file not in scale_config:
            missing_videos.append(video_file)
    
    return missing_videos


def print_cache_status(scale_config, missing_videos=None):
    """
    列印快取狀態摘要
    
    Args:
        scale_config: 比例尺配置字典
        missing_videos: 缺少的影片列表
    """
    print("📊 比例尺快取狀態:")
    print("-" * 50)
    
    if scale_config:
        print(f"✅ 已快取影片數量: {len(scale_config)}")
        for video, scale in sorted(scale_config.items()):
            print(f"  📹 {video}: {scale:.2f} 像素")
    else:
        print("⚠️  沒有快取資料")
    
    if missing_videos:
        print(f"\n🔄 需要計算的影片: {len(missing_videos)}")
        for video in missing_videos:
            print(f"  📹 {video}")
    
    print("-" * 50)
