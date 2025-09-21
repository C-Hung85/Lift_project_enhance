#!/usr/bin/env python3
"""
調試 cluster ID 分配問題
"""
import json
import pandas as pd

def debug_cluster_assignment():
    """調試群集分配問題"""
    
    # 讀取清理報告
    with open('lifts/result/1_cleaning_report.json', 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # 找出時戳 13.171 和 13.272 的處理詳情
    target_timestamps = [13.171, 13.272]
    
    print("🔍 調試時戳 13.171 和 13.272 的群集分配:")
    print("=" * 60)
    
    for detail in report['processing_details']:
        if abs(detail['timestamp'] - 13.171) < 0.001 or abs(detail['timestamp'] - 13.272) < 0.001:
            print(f"時戳: {detail['timestamp']:.3f}")
            print(f"值: {detail['original_value']:.6f}")
            print(f"動作: {detail['action']}")
            print(f"原因: {detail['reason']}")
            print(f"窗口大小: {detail.get('window_size', 'N/A')}")
            print(f"群集ID: {detail['cluster_id']}")
            print("-" * 40)
    
    # 檢查對應的群集分析
    print("\n🔍 對應的群集分析:")
    print("=" * 60)
    
    target_cluster_ids = set()
    for detail in report['processing_details']:
        if abs(detail['timestamp'] - 13.171) < 0.001 or abs(detail['timestamp'] - 13.272) < 0.001:
            target_cluster_ids.add(detail['cluster_id'])
    
    for cluster in report['cluster_analysis']:
        if cluster['cluster_id'] in target_cluster_ids:
            print(f"群集ID: {cluster['cluster_id']}")
            print(f"開始時戳: {cluster['start_timestamp']:.3f}")
            print(f"結束時戳: {cluster['end_timestamp']:.3f}")
            print(f"模式類型: {cluster['pattern_type']}")
            print(f"值數量: {cluster['total_values']}")
            print(f"清理前總和: {cluster['sum_before_cleaning']:.6f}")
            print(f"動作: {cluster['action']}")
            print("-" * 40)
    
    # 檢查原始數據
    print("\n🔍 原始數據檢查:")
    print("=" * 60)
    
    df = pd.read_csv('lifts/result/1.csv')
    
    # 找到這兩個時戳在原始數據中的位置
    for target_time in target_timestamps:
        time_diff = abs(df.iloc[:, 0] - target_time)
        closest_idx = time_diff.idxmin()
        
        print(f"時戳 {target_time:.3f} 的原始數據:")
        start_idx = max(0, closest_idx - 2)
        end_idx = min(len(df), closest_idx + 3)
        
        for i in range(start_idx, end_idx):
            marker = " ← 目標" if i == closest_idx else ""
            time_val = df.iloc[i, 0]
            disp_val = df.iloc[i, 1]
            print(f"  行{i+2:4d}: {time_val:8.3f}, {disp_val:12.6f}{marker}")
        print()

if __name__ == '__main__':
    debug_cluster_assignment()
