#!/usr/bin/env python3
"""
測試窗口檢測邏輯
"""
import numpy as np
from data_cleaner import DataCleaner

def test_alternating_pair():
    """測試交替對的檢測"""
    
    # 模擬包含 13.171 和 13.272 的數據
    timestamps = [13.071, 13.171, 13.272, 13.372, 13.473]
    values = [0.0, -0.11600928074245939, 0.11600928074245939, 0.0, 0.0]
    
    print("🧪 測試窗口檢測邏輯:")
    print("=" * 50)
    print("測試數據:")
    for i, (t, v) in enumerate(zip(timestamps, values)):
        print(f"  {i}: {t:.3f}s, {v:.6f}")
    
    cleaner = DataCleaner()
    
    # 測試 2-窗口
    print(f"\n🔍 測試窗口大小 = 2:")
    for i in range(len(values) - 1):
        window = values[i:i+2]
        window_times = timestamps[i:i+2]
        
        # 檢測雜訊閾值
        noise_threshold = cleaner.detect_noise_threshold(values)
        
        is_noise = cleaner.is_noise_pattern(window, noise_threshold)
        pattern_type = cleaner._classify_pattern(window)
        
        print(f"  窗口 [{i}:{i+2}]: {window}")
        print(f"    時戳: {window_times}")
        print(f"    是雜訊: {is_noise}")
        print(f"    模式: {pattern_type}")
        print(f"    總和: {sum(window):.6f}")
        print()
    
    # 執行完整清理
    print("🔄 執行完整清理:")
    print("=" * 50)
    
    cleaned_values, noise_count, threshold, basic_unit, processing_details, cluster_analysis = cleaner.clean_data(values, timestamps)
    
    print(f"基本單位: {basic_unit:.6f}")
    print(f"雜訊閾值: {threshold:.6f}")
    print(f"移除數量: {noise_count}")
    
    print(f"\n清理結果:")
    for i, (orig, clean) in enumerate(zip(values, cleaned_values)):
        status = "移除" if orig != clean else "保留"
        print(f"  {timestamps[i]:.3f}s: {orig:.6f} → {clean:.6f} ({status})")
    
    print(f"\n處理詳情:")
    for detail in processing_details:
        print(f"  時戳 {detail['timestamp']:.3f}: 動作={detail['action']}, 群集ID={detail.get('cluster_id', 'N/A')}")
    
    print(f"\n群集分析:")
    for cluster in cluster_analysis:
        print(f"  群集 {cluster['cluster_id']}: {cluster['start_timestamp']:.3f}s-{cluster['end_timestamp']:.3f}s, 模式={cluster['pattern_type']}")

if __name__ == '__main__':
    test_alternating_pair()
