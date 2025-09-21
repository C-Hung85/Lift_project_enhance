#!/usr/bin/env python3
"""
數據清理腳本 - 移除CSV檔案中的畫面抖動雜訊
"""
import os
import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class DataCleaner:
    def __init__(self):
        self.noise_threshold_multiplier = 2.0  # 雜訊閾值倍數，改為2.0倍更嚴格
        self.isolation_threshold_seconds = 1.0  # 離群值時間閾值（秒）
    
    def detect_noise_threshold(self, values):
        """
        檢測每個檔案的雜訊閾值
        基於最小非零絕對值來判斷基本畫素震盪大小
        考慮計算誤差的容忍度
        """
        values = np.array(values)
        non_zero = values[values != 0.0]
        
        if len(non_zero) == 0:
            return 0.0
        
        # 找到最小的非零絕對值作為基本雜訊單位
        abs_values = np.abs(non_zero)
        min_abs_value = np.min(abs_values)
        
        # 分析所有非零值，找出基本單位的近似值
        # 允許10%的計算誤差容忍度
        tolerance = 0.1
        basic_unit = min_abs_value
        
        # 檢查是否有其他值是基本單位的倍數（考慮誤差）
        possible_multiples = []
        for val in abs_values:
            ratio = val / basic_unit
            # 檢查是否接近整數倍
            rounded_ratio = round(ratio)
            if abs(ratio - rounded_ratio) <= tolerance and rounded_ratio <= 5:
                possible_multiples.append(rounded_ratio)
        
        # 如果大部分值都是基本單位的整數倍，確認基本單位
        if len(set(possible_multiples)) > 1:
            # 有多種倍數，基本單位應該正確
            pass
        else:
            # 可能需要重新調整基本單位
            # 尋找最大公約數的近似值
            sorted_values = np.sort(abs_values)
            if len(sorted_values) > 1:
                # 使用最小的兩個值來推估基本單位
                basic_unit = min(sorted_values[0], sorted_values[1] / round(sorted_values[1] / sorted_values[0]))
        
        # 雜訊閾值設為基本單位的倍數，加上容忍度
        noise_threshold = basic_unit * (self.noise_threshold_multiplier + tolerance)
        
        return noise_threshold
    
    def is_noise_pattern(self, window, noise_threshold):
        """
        判斷一個窗口是否為雜訊模式
        條件：
        1. 窗口內有非零值
        2. 窗口內所有值的絕對值都小於雜訊閾值
        3. 窗口的總和接近零（正負相消）
        4. 考慮計算誤差的容忍度
        """
        window = np.array(window)
        
        # 檢查是否有非零值
        if np.all(window == 0):
            return False
        
        # 檢查所有非零值是否都在雜訊閾值內
        non_zero_mask = window != 0
        if np.any(np.abs(window[non_zero_mask]) > noise_threshold):
            return False
        
        # 檢查窗口總和是否接近零（相對於雜訊閾值）
        # 放寬容忍度以處理計算誤差
        window_sum = np.sum(window)
        tolerance_factor = 0.8  # 允許更大的總和偏差
        if abs(window_sum) > noise_threshold * tolerance_factor:
            return False
        
        # 額外檢查：如果窗口內有正負值交替模式，更可能是雜訊
        non_zero_values = window[non_zero_mask]
        if len(non_zero_values) >= 2:
            # 檢查是否有正負值
            has_positive = np.any(non_zero_values > 0)
            has_negative = np.any(non_zero_values < 0)
            
            if has_positive and has_negative:
                # 有正負值，更可能是雜訊，降低閾值要求
                return abs(window_sum) <= noise_threshold * 1.2
        
        return True
    
    def clean_data(self, values, timestamps=None):
        """
        清理數據中的雜訊 - 新的兩階段方法
        階段一：Window Size 2 正負對檢測
        階段二：離群單獨值檢測
        返回清理後的數據和詳細的處理記錄
        """
        values = np.array(values, dtype=float)
        original_values = values.copy()
        cleaned_values = values.copy()
        
        if timestamps is None:
            timestamps = list(range(len(values)))
        
        # 檢測雜訊閾值和基本單位
        noise_threshold = self.detect_noise_threshold(values)
        basic_unit = self._get_basic_unit(values)
        
        # 初始化處理記錄
        processing_details = []
        cluster_analysis = []
        cluster_id = 0
        
        # 階段統計
        stage1_stats = {"pairs_detected": 0, "points_removed": 0, "positive_negative": 0, "negative_positive": 0}
        stage2_stats = {"candidates_checked": 0, "points_removed": 0}
        
        if noise_threshold == 0:
            # 沒有非零值，記錄所有零值
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                if value == 0:
                    processing_details.append({
                        "timestamp": timestamp,
                        "original_value": value,
                        "action": "retained",
                        "stage": "final",
                        "reason": "zero_value",
                        "cluster_id": None
                    })
            return cleaned_values, 0, noise_threshold, basic_unit, processing_details, cluster_analysis, stage1_stats, stage2_stats
        
        # === 階段一：Window Size 2 正負對檢測 ===
        processed_indices = set()
        i = 0
        while i <= len(cleaned_values) - 2:
            # 檢查是否已處理
            if i in processed_indices or (i + 1) in processed_indices:
                i += 1
                continue
            
            window = cleaned_values[i:i+2]
            window_timestamps = timestamps[i:i+2]
            
            if self._is_noise_pair(window, noise_threshold):
                cluster_id += 1
                stage1_stats["pairs_detected"] += 1
                stage1_stats["points_removed"] += 2
                
                # 判斷是正負對還是負正對
                if window[0] > 0 and window[1] < 0:
                    pair_type = "positive_negative"
                    stage1_stats["positive_negative"] += 1
                else:
                    pair_type = "negative_positive"
                    stage1_stats["negative_positive"] += 1
                
                # 記錄群集分析
                cluster_analysis.append({
                    "cluster_id": cluster_id,
                    "stage": "stage_1_pair",
                    "pattern_type": f"{pair_type}_pair",
                    "start_timestamp": window_timestamps[0],
                    "end_timestamp": window_timestamps[1],
                    "total_values": 2,
                    "sum_before_cleaning": float(np.sum(window)),
                    "action": "removed_as_noise"
                })
                
                # 記錄每個值的處理詳情
                for j, (timestamp, original_val) in enumerate(zip(window_timestamps, window)):
                    processing_details.append({
                        "timestamp": timestamp,
                        "original_value": original_val,
                        "action": "removed",
                        "stage": "stage_1_pair",
                        "reason": f"{pair_type}_pair",
                        "pair_partner_timestamp": window_timestamps[1-j],  # 配對夥伴的時間戳
                        "cluster_id": cluster_id
                    })
                    processed_indices.add(i + j)
                
                # 將配對設為0
                cleaned_values[i:i+2] = 0.0
                i += 2  # 跳過已處理的配對
            else:
                i += 1
        
        # === 階段二：離群單獨值檢測 ===
        for i, (timestamp, original_val, cleaned_val) in enumerate(zip(timestamps, original_values, cleaned_values)):
            if i not in processed_indices and cleaned_val != 0:
                stage2_stats["candidates_checked"] += 1
                
                if self._is_isolated_noise(i, timestamps, cleaned_values, basic_unit):
                    cluster_id += 1
                    stage2_stats["points_removed"] += 1
                    
                    # 記錄群集分析
                    cluster_analysis.append({
                        "cluster_id": cluster_id,
                        "stage": "stage_2_isolated",
                        "pattern_type": "isolated_small_value",
                        "start_timestamp": timestamp,
                        "end_timestamp": timestamp,
                        "total_values": 1,
                        "sum_before_cleaning": float(cleaned_val),
                        "action": "removed_as_noise"
                    })
                    
                    # 記錄處理詳情
                    processing_details.append({
                        "timestamp": timestamp,
                        "original_value": original_val,
                        "action": "removed",
                        "stage": "stage_2_isolated",
                        "reason": "isolated_small_value",
                        "cluster_id": cluster_id
                    })
                    
                    # 移除離群值
                    cleaned_values[i] = 0.0
                    processed_indices.add(i)
        
        # === 記錄保留的值 ===
        for i, (timestamp, original_val, cleaned_val) in enumerate(zip(timestamps, original_values, cleaned_values)):
            if i not in processed_indices and original_val != 0:
                if cleaned_val != 0:  # 被保留的值
                    cluster_id += 1
                    
                    # 記錄群集分析
                    cluster_analysis.append({
                        "cluster_id": cluster_id,
                        "stage": "final",
                        "pattern_type": "valid_movement",
                        "start_timestamp": timestamp,
                        "end_timestamp": timestamp,
                        "total_values": 1,
                        "sum_before_cleaning": float(original_val),
                        "action": "retained_as_valid"
                    })
                    
                    # 記錄處理詳情
                    processing_details.append({
                        "timestamp": timestamp,
                        "original_value": original_val,
                        "action": "retained",
                        "stage": "final",
                        "reason": "valid_movement",
                        "cluster_id": cluster_id
                    })
        
        # 計算移除的雜訊點數
        original_non_zero = np.sum(original_values != 0)
        cleaned_non_zero = np.sum(cleaned_values != 0)
        total_noise_removed = original_non_zero - cleaned_non_zero
        
        return cleaned_values, total_noise_removed, noise_threshold, basic_unit, processing_details, cluster_analysis, stage1_stats, stage2_stats
    
    def _get_basic_unit(self, values):
        """獲取基本雜訊單位"""
        values = np.array(values)
        non_zero = values[values != 0.0]
        
        if len(non_zero) == 0:
            return 0.0
        
        abs_values = np.abs(non_zero)
        return float(np.min(abs_values))
    
    def _is_noise_pair(self, window, noise_threshold):
        """
        判斷一個大小為2的窗口是否為雜訊配對
        條件：
        1. 兩個值都非零
        2. 兩個值符號相反（一正一負）
        3. 兩個值的絕對值都小於雜訊閾值
        4. 兩個值的總和接近零（正負相消）
        """
        window = np.array(window)
        
        # 檢查是否兩個值都非零
        if np.any(window == 0):
            return False
        
        # 檢查是否符號相反
        if window[0] * window[1] >= 0:  # 同號或其中一個為0
            return False
        
        # 檢查兩個值的絕對值是否都在雜訊閾值內
        if np.any(np.abs(window) > noise_threshold):
            return False
        
        # 檢查總和是否接近零（正負相消）
        window_sum = np.sum(window)
        tolerance_factor = 0.3  # 允許30%的偏差
        return abs(window_sum) <= noise_threshold * tolerance_factor
    
    def _is_isolated_noise(self, index, timestamps, values, basic_unit):
        """
        判斷是否為離群雜訊
        條件：
        1. 值為1-2.1倍基本單位大小（包含10%容忍度）
        2. 與最近的非零值相隔超過 isolation_threshold_seconds 秒
        """
        current_value = values[index]
        current_time = timestamps[index]
        
        # 檢查是否為小值（1-2.1倍基本單位，與階段一保持一致）
        tolerance = 0.1  # 10%容忍度
        max_threshold = basic_unit * (self.noise_threshold_multiplier + tolerance)
        if abs(current_value) > max_threshold:
            return False
        
        # 找尋最近的非零值的時間距離
        min_distance = float('inf')
        for i, val in enumerate(values):
            if i != index and val != 0:
                distance = abs(timestamps[i] - current_time)
                min_distance = min(min_distance, distance)
        
        # 如果沒有其他非零值，也算是離群
        if min_distance == float('inf'):
            return True
        
        return min_distance > self.isolation_threshold_seconds
    
    def _classify_pattern(self, window):
        """分類雜訊模式"""
        window = np.array(window)
        non_zero = window[window != 0]
        
        if len(non_zero) == 0:
            return "all_zero"
        elif len(non_zero) == 1:
            return "single_value"
        elif len(non_zero) == 2:
            if non_zero[0] * non_zero[1] < 0:  # 相反符號
                return "alternating_pair"
            else:
                return "same_sign_pair"
        elif len(non_zero) == 4:
            signs = np.sign(non_zero)
            if np.array_equal(signs, [1, -1, -1, 1]) or np.array_equal(signs, [-1, 1, 1, -1]):
                return "four_alternating_pattern"
            elif np.array_equal(signs, [1, -1, 1, -1]) or np.array_equal(signs, [-1, 1, -1, 1]):
                return "strict_alternating_pattern"
            else:
                return "four_value_pattern"
        else:
            return f"{len(non_zero)}_value_pattern"
    
    def process_csv_file(self, input_path, output_path):
        """
        處理單個CSV檔案並生成詳細報告
        """
        times = []
        values = []
        
        # 讀取數據
        with open(input_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # 讀取標題行
            
            for row in reader:
                if len(row) >= 2:
                    try:
                        time_val = float(row[0])
                        displacement_val = float(row[1])
                        times.append(time_val)
                        values.append(displacement_val)
                    except ValueError:
                        continue
        
        # 清理數據並獲取詳細記錄
        cleaned_values, noise_count, threshold, basic_unit, processing_details, cluster_analysis, stage1_stats, stage2_stats = self.clean_data(values, times)
        
        # 寫入清理後的數據
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)  # 寫入標題行
            
            for time_val, clean_val in zip(times, cleaned_values):
                writer.writerow([time_val, clean_val])
        
        # 生成清理報告
        report = self._generate_cleaning_report(
            input_path, output_path, times, values, cleaned_values,
            basic_unit, threshold, processing_details, cluster_analysis, stage1_stats, stage2_stats
        )
        
        # 保存報告
        report_path = self._save_cleaning_report(input_path, report)
        
        return {
            'original_points': len(values),
            'noise_points_removed': noise_count,
            'noise_threshold': threshold,
            'non_zero_original': np.sum(np.array(values) != 0),
            'non_zero_cleaned': np.sum(cleaned_values != 0),
            'report_path': report_path
        }
    
    def _generate_cleaning_report(self, input_path, output_path, times, original_values, cleaned_values,
                                 basic_unit, threshold, processing_details, cluster_analysis, stage1_stats, stage2_stats):
        """生成清理報告 - 新的兩階段格式"""
        input_file = Path(input_path).name
        output_file = Path(output_path).name
        
        # 排序處理詳情
        processing_details.sort(key=lambda x: x['timestamp'])
        
        # 計算統計數據
        original_non_zero = np.sum(np.array(original_values) != 0)
        cleaned_non_zero = np.sum(cleaned_values != 0)
        total_removed = original_non_zero - cleaned_non_zero
        
        # 計算移除率
        total_removal_rate = (total_removed / original_non_zero * 100) if original_non_zero > 0 else 0
        stage1_removal_rate = (stage1_stats["points_removed"] / original_non_zero * 100) if original_non_zero > 0 else 0
        stage2_removal_rate = (stage2_stats["points_removed"] / original_non_zero * 100) if original_non_zero > 0 else 0
        
        report = {
            "file_info": {
                "original_file": input_file,
                "cleaned_file": output_file,
                "processing_time": datetime.now().isoformat(),
                "cleaner_version": "2.0"  # 新版本
            },
            "detection_parameters": {
                "basic_unit_mm": float(basic_unit),
                "noise_threshold_mm": float(threshold),
                "threshold_multiplier": float(self.noise_threshold_multiplier),
                "isolation_threshold_seconds": float(self.isolation_threshold_seconds),
                "tolerance_factor": 0.1
            },
            "processing_stages": {
                "stage_1_pair_detection": {
                    "description": "Window Size 2 正負對檢測",
                    "pairs_detected": stage1_stats["pairs_detected"],
                    "points_removed": stage1_stats["points_removed"],
                    "pattern_types": {
                        "positive_negative": stage1_stats["positive_negative"],
                        "negative_positive": stage1_stats["negative_positive"]
                    }
                },
                "stage_2_isolated_detection": {
                    "description": "離群單獨值檢測",
                    "candidates_checked": stage2_stats["candidates_checked"],
                    "points_removed": stage2_stats["points_removed"],
                    "isolation_criteria": {
                        "min_distance_seconds": float(self.isolation_threshold_seconds),
                        "max_value_multiplier": 2.1  # 2.0 + 10%容忍度
                    }
                }
            },
            "summary_stats": {
                "total_data_points": len(original_values),
                "original_non_zero_points": int(original_non_zero),
                "stage_1_removed": int(stage1_stats["points_removed"]),
                "stage_2_removed": int(stage2_stats["points_removed"]),
                "total_removed": int(total_removed),
                "final_non_zero_points": int(cleaned_non_zero),
                "total_removal_rate": round(total_removal_rate, 1),
                "stage_1_removal_rate": round(stage1_removal_rate, 1),
                "stage_2_removal_rate": round(stage2_removal_rate, 1)
            },
            "processing_details": processing_details,
            "cluster_analysis": cluster_analysis
        }
        
        return report
    
    def _save_cleaning_report(self, input_path, report):
        """保存清理報告到JSON檔案"""
        input_path = Path(input_path)
        
        # 生成報告檔名：原檔名_cleaning_report.json
        report_filename = f"{input_path.stem}_cleaning_report.json"
        report_path = input_path.parent / report_filename
        
        # 保存為格式化的JSON
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(report_path)
    
    def process_directory(self, input_dir, output_dir=None):
        """
        處理目錄中的所有CSV檔案
        """
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_path = input_path
        else:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        # 只處理原始檔案，不處理已清理的檔案（避免重複處理c*.csv檔案）
        csv_files = [f for f in input_path.glob("*.csv") if not f.name.startswith('c')]
        
        if not csv_files:
            print(f"在 {input_dir} 中沒有找到 CSV 檔案")
            return
        
        print(f"找到 {len(csv_files)} 個 CSV 檔案\n")
        
        total_stats = {
            'files_processed': 0,
            'total_noise_removed': 0,
            'total_original_points': 0
        }
        
        for csv_file in csv_files:
            # 生成輸出檔名（加上 'c' 前綴）
            output_filename = f"c{csv_file.name}"
            output_file_path = output_path / output_filename
            
            print(f"處理: {csv_file.name} -> {output_filename}")
            
            try:
                stats = self.process_csv_file(csv_file, output_file_path)
                
                print(f"  原始數據點: {stats['original_points']}")
                print(f"  原始非零點: {stats['non_zero_original']}")
                print(f"  清理後非零點: {stats['non_zero_cleaned']}")
                print(f"  總移除雜訊點: {stats['noise_points_removed']}")
                print(f"  雜訊閾值: {stats['noise_threshold']:.6f} mm")
                
                # 分階段統計（從報告中讀取）
                report_path = stats['report_path']
                try:
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    stage1_removed = report_data['summary_stats']['stage_1_removed']
                    stage2_removed = report_data['summary_stats']['stage_2_removed']
                    pairs_detected = report_data['processing_stages']['stage_1_pair_detection']['pairs_detected']
                    
                    print(f"  📊 階段一 (正負對): 檢測到 {pairs_detected} 對, 移除 {stage1_removed} 點")
                    print(f"  📊 階段二 (離群值): 移除 {stage2_removed} 點")
                    
                    # 安全計算雜訊移除率
                    if stats['non_zero_original'] > 0:
                        total_rate = stats['noise_points_removed']/stats['non_zero_original']*100
                        stage1_rate = stage1_removed/stats['non_zero_original']*100
                        stage2_rate = stage2_removed/stats['non_zero_original']*100
                        print(f"  📈 總移除率: {total_rate:.1f}% (階段一: {stage1_rate:.1f}%, 階段二: {stage2_rate:.1f}%)")
                    else:
                        print(f"  📈 總移除率: 0.0%")
                except:
                    # 如果讀取報告失敗，使用舊格式
                    if stats['non_zero_original'] > 0:
                        removal_rate = stats['noise_points_removed']/stats['non_zero_original']*100
                        print(f"  雜訊移除率: {removal_rate:.1f}%")
                    else:
                        print(f"  雜訊移除率: 0.0%")
                
                # 顯示報告路徑
                print(f"  📊 清理報告: {Path(stats['report_path']).name}")
                
                total_stats['files_processed'] += 1
                total_stats['total_noise_removed'] += stats['noise_points_removed']
                total_stats['total_original_points'] += stats['original_points']
                
            except Exception as e:
                print(f"  錯誤: {e}")
            
            print()
        
        print("=" * 50)
        print("總處理統計:")
        print(f"處理檔案數: {total_stats['files_processed']}")
        print(f"總數據點: {total_stats['total_original_points']}")
        print(f"總雜訊移除數: {total_stats['total_noise_removed']}")


def main():
    """
    主函數 - 處理 lifts/result 目錄中的所有 CSV 檔案
    """
    cleaner = DataCleaner()
    
    # 處理 lifts/result 目錄
    input_directory = "lifts/result"
    
    if not os.path.exists(input_directory):
        print(f"錯誤: 目錄 {input_directory} 不存在")
        return
    
    print("數據清理工具 - 移除畫面抖動雜訊")
    print("=" * 50)
    
    cleaner.process_directory(input_directory)
    
    print("清理完成！")


if __name__ == '__main__':
    main()
