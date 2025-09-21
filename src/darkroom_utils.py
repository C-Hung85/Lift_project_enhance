"""
暗房時間區間處理工具函數模組
"""
import re


def parse_time_string(time_str):
    """
    解析時間字串 'MM:SS' 轉換為秒數
    
    Args:
        time_str: 時間字串，格式為 'MM:SS' (分:秒)
    
    Returns:
        seconds: 總秒數
    """
    try:
        # 支援 MM:SS 格式
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
        
        # 如果只是數字，當作秒數
        return int(float(time_str))
    
    except (ValueError, AttributeError):
        raise ValueError(f"無法解析時間格式: {time_str}，請使用 'MM:SS' 格式")


def convert_intervals_to_seconds(intervals):
    """
    將時間區間列表轉換為秒數格式
    
    Args:
        intervals: [('開始時間', '結束時間'), ...] 字串格式
    
    Returns:
        intervals_seconds: [(開始秒數, 結束秒數), ...] 數字格式
    """
    intervals_seconds = []
    
    for start_str, end_str in intervals:
        try:
            start_seconds = parse_time_string(start_str)
            end_seconds = parse_time_string(end_str)
            
            if start_seconds >= end_seconds:
                raise ValueError(f"開始時間 {start_str} 必須小於結束時間 {end_str}")
            
            intervals_seconds.append((start_seconds, end_seconds))
        
        except ValueError as e:
            print(f"⚠️  時間區間格式錯誤: {e}")
            continue
    
    return intervals_seconds


def is_in_darkroom_interval(current_time_seconds, intervals_seconds):
    """
    檢查當前時間是否在暗房區間內
    
    Args:
        current_time_seconds: 當前時間（秒）
        intervals_seconds: 暗房時間區間列表 [(開始秒數, 結束秒數), ...]
    
    Returns:
        is_darkroom: 是否在暗房區間內
        interval_info: 如果在區間內，返回區間資訊，否則為 None
    """
    for start_seconds, end_seconds in intervals_seconds:
        if start_seconds <= current_time_seconds <= end_seconds:
            return True, (start_seconds, end_seconds)
    
    return False, None


def format_time_seconds(seconds):
    """
    將秒數格式化為時間字串
    
    Args:
        seconds: 秒數
    
    Returns:
        time_str: 格式化的時間字串 'MM:SS'
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def get_darkroom_intervals_for_video(filename, darkroom_config):
    """
    取得特定影片的暗房時間區間
    
    Args:
        filename: 影片檔名
        darkroom_config: 暗房配置字典
    
    Returns:
        intervals_seconds: 該影片的暗房時間區間（秒數格式）
        has_darkroom: 是否有暗房區間設定
    """
    if filename not in darkroom_config:
        return [], False
    
    intervals = darkroom_config[filename]
    intervals_seconds = convert_intervals_to_seconds(intervals)
    
    return intervals_seconds, len(intervals_seconds) > 0


def validate_darkroom_config(darkroom_config):
    """
    驗證暗房配置的有效性
    
    Args:
        darkroom_config: 暗房配置字典
    
    Returns:
        is_valid: 配置是否有效
        errors: 錯誤訊息列表
    """
    errors = []
    
    for filename, intervals in darkroom_config.items():
        if not filename.endswith('.mp4'):
            errors.append(f"檔名格式錯誤: {filename}，應以 .mp4 結尾")
            continue
        
        if not isinstance(intervals, list):
            errors.append(f"{filename}: 區間設定應為列表格式")
            continue
        
        for i, interval in enumerate(intervals):
            if not isinstance(interval, tuple) or len(interval) != 2:
                errors.append(f"{filename}: 區間 {i+1} 格式錯誤，應為 ('開始時間', '結束時間')")
                continue
            
            start_str, end_str = interval
            try:
                start_seconds = parse_time_string(start_str)
                end_seconds = parse_time_string(end_str)
                
                if start_seconds >= end_seconds:
                    errors.append(f"{filename}: 區間 {i+1} 開始時間 {start_str} 必須小於結束時間 {end_str}")
                
            except ValueError as e:
                errors.append(f"{filename}: 區間 {i+1} 時間格式錯誤: {e}")
    
    return len(errors) == 0, errors


def print_darkroom_summary(darkroom_config):
    """
    列印暗房配置摘要
    
    Args:
        darkroom_config: 暗房配置字典
    """
    print("📑 暗房區間設定摘要:")
    print("=" * 50)
    
    if not darkroom_config:
        print("⚠️  沒有設定任何暗房區間")
        return
    
    for filename, intervals in darkroom_config.items():
        print(f"\n🎬 {filename}:")
        
        if not intervals:
            print("  (無暗房區間)")
            continue
        
        for i, (start_str, end_str) in enumerate(intervals, 1):
            try:
                start_seconds = parse_time_string(start_str)
                end_seconds = parse_time_string(end_str)
                duration = end_seconds - start_seconds
                
                print(f"  🌙 區間 {i}: {start_str} ~ {end_str} (持續 {format_time_seconds(duration)})")
                
            except ValueError as e:
                print(f"  ❌ 區間 {i}: {start_str} ~ {end_str} (格式錯誤: {e})")
    
    print("\n" + "=" * 50)
