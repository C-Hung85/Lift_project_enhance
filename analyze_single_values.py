#!/usr/bin/env python3
"""
分析清理報告中的 single_value 模式
"""
import json
import pandas as pd

def analyze_single_values(report_file, csv_file, num_samples=5):
    """
    分析 single_value 模式的時戳和周圍數據
    
    Args:
        report_file: 清理報告 JSON 檔案路徑
        csv_file: 原始 CSV 檔案路徑
        num_samples: 要分析的樣本數量
    """
    # 讀取清理報告
    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    # 讀取原始 CSV 數據
    df = pd.read_csv(csv_file)
    
    # 找出所有 single_value 的群集
    single_value_clusters = [
        cluster for cluster in report['cluster_analysis'] 
        if cluster['pattern_type'] == 'single_value'
    ]
    
    print(f"📊 發現 {len(single_value_clusters)} 個 single_value 群集")
    print(f"🔍 分析前 {num_samples} 個樣本:\n")
    
    # 分析前 num_samples 個
    for i, cluster in enumerate(single_value_clusters[:num_samples]):
        timestamp = cluster['start_timestamp']
        value = cluster['sum_before_cleaning']
        
        print(f"=== 樣本 {i+1} ===")
        print(f"時戳: {timestamp:.3f} 秒")
        print(f"位移值: {value:.6f} mm")
        print(f"動作: {cluster['action']}")
        
        # 在 CSV 中找到對應的行
        time_col = df.columns[0]  # 第一欄是時間
        disp_col = df.columns[1]  # 第二欄是位移
        
        # 找到最接近的時戳索引
        time_diff = abs(df[time_col] - timestamp)
        closest_idx = time_diff.idxmin()
        
        # 顯示周圍的上下文（前後各3行）
        start_idx = max(0, closest_idx - 3)
        end_idx = min(len(df), closest_idx + 4)
        
        print(f"周圍數據 (行 {start_idx+2} 到 {end_idx+1}):")
        print("   行號  |   時戳    |    位移值     | 標記")
        print("-" * 45)
        
        for idx in range(start_idx, end_idx):
            row_num = idx + 2  # CSV 行號（從2開始，因為有標題行）
            time_val = df.iloc[idx][time_col]
            disp_val = df.iloc[idx][disp_col]
            
            # 標記目標行
            marker = " ← 目標" if idx == closest_idx else ""
            print(f"  {row_num:5d}  | {time_val:8.3f} | {disp_val:12.6f} {marker}")
        
        print()
    
    # 統計 single_value 的分佈
    print("📈 Single Value 統計分析:")
    values = [cluster['sum_before_cleaning'] for cluster in single_value_clusters]
    
    print(f"  總數量: {len(values)}")
    print(f"  最小值: {min(values):.6f} mm")
    print(f"  最大值: {max(values):.6f} mm")
    print(f"  平均值: {sum(values)/len(values):.6f} mm")
    
    # 分析保留 vs 移除的比例
    retained = sum(1 for cluster in single_value_clusters if cluster['action'] == 'retained_as_valid')
    removed = len(single_value_clusters) - retained
    
    print(f"  保留: {retained} 個 ({retained/len(single_value_clusters)*100:.1f}%)")
    print(f"  移除: {removed} 個 ({removed/len(single_value_clusters)*100:.1f}%)")
    
    # 顯示值的分佈
    print(f"\n📊 數值分佈:")
    unique_values = {}
    for val in values:
        rounded_val = round(val, 6)
        unique_values[rounded_val] = unique_values.get(rounded_val, 0) + 1
    
    # 顯示最常見的值
    sorted_values = sorted(unique_values.items(), key=lambda x: x[1], reverse=True)
    print("  最常見的 single_value 數值:")
    for val, count in sorted_values[:10]:  # 顯示前10個
        print(f"    {val:12.6f} mm: {count:3d} 次 ({count/len(values)*100:.1f}%)")

def main():
    """主函數"""
    report_file = "lifts/result/1_cleaning_report.json"
    csv_file = "lifts/result/1.csv"
    
    print("🔍 Single Value 模式分析工具")
    print("=" * 50)
    
    try:
        analyze_single_values(report_file, csv_file, num_samples=5)
    except FileNotFoundError as e:
        print(f"❌ 檔案不存在: {e}")
    except Exception as e:
        print(f"❌ 分析失敗: {e}")

if __name__ == '__main__':
    main()
