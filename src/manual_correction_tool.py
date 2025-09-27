#!/usr/bin/env python3
"""
半自動人工校正工具
用於手動校正電梯位移檢測數據
"""
import sys
import os
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json
from datetime import datetime
import argparse

# 添加 src 目錄到路徑以導入配置模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from rotation_config import rotation_config
except ImportError:
    rotation_config = {}

try:
    from scale_config import scale_config
except ImportError:
    scale_config = {}

from rotation_utils import rotate_frame

# --- Helper: optional affine frame mapping (main_frame -> source_frame) ---
def map_frame_index(main_frame_index: int, intercept: float, slope: float) -> int:
    """將主程式/CSV 幀號映射為原檔時基幀號。

    Args:
        main_frame_index: 來自 CSV/inspection 的幀號
        intercept: 仿射截距（預設建議 318）
        slope: 仿射斜率（預設建議 0.9946）
    Returns:
        以原檔時間軸估計的整數幀號
    """
    mapped = intercept + slope * float(main_frame_index)
    return int(round(mapped))

@dataclass
class CorrectionCluster:
    """位移校正群集數據結構"""
    start_index: int              # 群集開始的索引
    end_index: int                # 群集結束的索引
    pre_zero_index: int           # 群集前的零點索引
    timestamps: List[float]       # 對應的時戳 [pre_zero, start, ..., end]
    frame_indices: List[int]      # 對應的幀號 [pre_zero, start, ..., end]
    original_values: List[float]  # 原始位移值 [start, ..., end]
    csv_indices: List[int]        # CSV中的行號 [pre_zero, start, ..., end]

@dataclass
class PhysicalCluster:
    """物理群集數據結構"""
    cluster_id: int                    # 物理群集序號
    pre_zero_index: int                # 前0點CSV行號
    post_zero_index: int               # 後0點CSV行號
    pre_zero_jpg: str                  # 前0點JPG檔名
    post_zero_jpg: str                 # 後0點JPG檔名
    region_values: List[float]         # 區間內的所有位移值
    is_pure_noise: bool                # 是否為純雜訊群集（區間內全為0）
    has_real_motion: bool              # 是否包含真實運動

@dataclass
class ReferenceLine:
    """參考線段數據結構"""
    timestamp: float
    start_pixel_coords: Tuple[int, int]  # 線段起點 (x, y) 在原始影片中的座標
    end_pixel_coords: Tuple[int, int]    # 線段終點 (x, y) 在原始影片中的座標
    csv_index: int
    start_roi_coords: Tuple[int, int]    # 線段起點在ROI中的座標
    end_roi_coords: Tuple[int, int]      # 線段終點在ROI中的座標
    
    @property
    def y_component(self) -> float:
        """計算線段的Y分量長度"""
        return abs(self.end_pixel_coords[1] - self.start_pixel_coords[1])
    
    @property 
    def length(self) -> float:
        """計算線段總長度"""
        dx = self.end_pixel_coords[0] - self.start_pixel_coords[0]
        dy = self.end_pixel_coords[1] - self.start_pixel_coords[1]
        return (dx**2 + dy**2) ** 0.5

@dataclass
class ReferencePoint:
    """參考點數據結構 (保持向後兼容)"""
    timestamp: float
    pixel_coords: Tuple[int, int]  # (x, y) 在原始影片中的座標
    csv_index: int
    roi_coords: Tuple[int, int]    # (x, y) 在ROI中的座標

class DataManager:
    """數據管理模組"""
    
    def __init__(self, csv_path: str, video_name: str):
        self.csv_path = csv_path
        self.video_name = video_name
        self.df = pd.read_csv(csv_path)
        self.scale_factor = scale_config.get(video_name, None)
        
        # 檢查 'frame_idx' 欄位是否存在以提供向下相容性
        self.use_frame_indices = 'frame_idx' in self.df.columns
        if self.use_frame_indices:
            print("偵測到 'frame_idx' 欄位，將使用幀號進行精確提取。")
        else:
            print("⚠️ 警告: CSV 中未找到 'frame_idx' 欄位。將退回使用時間戳進行估算，可能會有偏差。")

        if self.scale_factor is None:
            raise ValueError(f"找不到影片 {video_name} 的比例尺配置")

        # 檢查是否有 frame_path 欄位（新的物理群集標籤系統）
        self.has_frame_path = 'frame_path' in self.df.columns
        if self.has_frame_path:
            print("✅ 偵測到 'frame_path' 欄位，使用物理群集標籤系統。")
            self.physical_clusters = self._identify_physical_clusters_from_png_tags()
            self.clusters = self._convert_physical_to_correction_clusters()
        else:
            print("⚠️ 使用舊版群集識別系統。")
            self.physical_clusters = []
            self.clusters = self._identify_clusters()
        
    def _identify_clusters(self) -> List[CorrectionCluster]:
        """識別所有需要校正的非零值群集"""
        clusters = []
        # 根據是否存在 frame_idx 欄位來決定 displacement_col 的索引
        if self.use_frame_indices:
            displacement_col = self.df.columns[2] # frame_idx, second, displacement
        else:
            displacement_col = self.df.columns[1]  # second, displacement
        
        i = 0
        while i < len(self.df):
            # 找到非零值
            if self.df.iloc[i][displacement_col] != 0:
                # 找到群集開始
                start_idx = i
                
                # 檢查是否有前零點可用
                if i > 0:
                    pre_zero_idx = i - 1
                    has_pre_zero = True
                else:
                    # 第一行就有位移，沒有前零點
                    pre_zero_idx = 0
                    has_pre_zero = False
                    print(f"警告: 檔案從第一行就開始有位移，將使用第一行作為參考點")
                
                # 找到群集結束
                while i < len(self.df) and self.df.iloc[i][displacement_col] != 0:
                    i += 1
                end_idx = i - 1
                
                # 建立時戳和幀號列表
                if has_pre_zero:
                    timestamps = [
                        self.df.iloc[pre_zero_idx]['second'],
                        *[self.df.iloc[j]['second'] for j in range(start_idx, end_idx + 1)]
                    ]
                    frame_indices = [
                        self.df.iloc[pre_zero_idx]['frame_idx'],
                        *[self.df.iloc[j]['frame_idx'] for j in range(start_idx, end_idx + 1)]
                    ] if self.use_frame_indices else []
                    csv_indices = list(range(pre_zero_idx, end_idx + 1))
                else:
                    timestamps = [self.df.iloc[j]['second'] for j in range(start_idx, end_idx + 1)]
                    frame_indices = [self.df.iloc[j]['frame_idx'] for j in range(start_idx, end_idx + 1)] if self.use_frame_indices else []
                    csv_indices = list(range(start_idx, end_idx + 1))
                
                # 建立群集
                cluster = CorrectionCluster(
                    start_index=start_idx,
                    end_index=end_idx,
                    pre_zero_index=pre_zero_idx,
                    timestamps=timestamps,
                    frame_indices=frame_indices,
                    original_values=[
                        self.df.iloc[j][displacement_col] for j in range(start_idx, end_idx + 1)
                    ],
                    csv_indices=csv_indices
                )
                
                # 為特殊情況添加標記
                setattr(cluster, 'has_pre_zero', has_pre_zero)
                
                clusters.append(cluster)
            else:
                i += 1
                
        return clusters

    def _identify_physical_clusters_from_png_tags(self) -> List[PhysicalCluster]:
        """基於PNG標籤識別物理群集 - 極其簡化的邏輯"""
        physical_clusters = []

        # 尋找所有前0點標籤
        for i, row in self.df.iterrows():
            frame_path = row.get('frame_path', '')

            if frame_path.startswith('pre_cluster_'):
                # 提取群集序號
                cluster_id = int(frame_path.split('_')[2].split('.')[0])

                # 找到對應的後0點
                post_tag = f'post_cluster_{cluster_id:03d}.jpg'
                post_rows = self.df[self.df['frame_path'] == post_tag]

                if not post_rows.empty:
                    pre_zero_index = i
                    post_zero_index = post_rows.index[0]

                    # 分析區間內的運動值
                    displacement_col = self.df.columns[2]  # displacement column
                    region_values = self.df.iloc[pre_zero_index:post_zero_index+1][displacement_col].tolist()

                    # 檢查是否為純雜訊群集
                    non_zero_values = [v for v in region_values if v != 0]
                    is_pure_noise = len(non_zero_values) == 0
                    has_real_motion = not is_pure_noise

                    cluster = PhysicalCluster(
                        cluster_id=cluster_id,
                        pre_zero_index=pre_zero_index,
                        post_zero_index=post_zero_index,
                        pre_zero_jpg=frame_path,
                        post_zero_jpg=post_tag,
                        region_values=region_values,
                        is_pure_noise=is_pure_noise,
                        has_real_motion=has_real_motion
                    )

                    # 只加入有真實運動的群集到校正清單
                    if has_real_motion:
                        physical_clusters.append(cluster)
                        print(f"✅ 識別物理群集 {cluster_id}：包含 {len(non_zero_values)} 個運動點")
                    else:
                        print(f"⚠️  跳過純雜訊群集 {cluster_id}：區間內無真實運動")

        print(f"📊 總共識別 {len(physical_clusters)} 個需要校正的物理群集")
        return physical_clusters

    def _convert_physical_to_correction_clusters(self) -> List[CorrectionCluster]:
        """將物理群集轉換為校正群集格式（向後兼容）"""
        correction_clusters = []

        for phys_cluster in self.physical_clusters:
            # 找到區間內的非零值範圍
            displacement_col = self.df.columns[2]
            non_zero_indices = []

            for i in range(phys_cluster.pre_zero_index, phys_cluster.post_zero_index + 1):
                if self.df.iloc[i][displacement_col] != 0:
                    non_zero_indices.append(i)

            if not non_zero_indices:
                continue

            start_idx = min(non_zero_indices)
            end_idx = max(non_zero_indices)

            # 建立時戳和幀號列表
            timestamps = [
                self.df.iloc[phys_cluster.pre_zero_index]['second'],
                *[self.df.iloc[j]['second'] for j in range(start_idx, end_idx + 1)]
            ]

            frame_indices = [
                self.df.iloc[phys_cluster.pre_zero_index]['frame_idx'],
                *[self.df.iloc[j]['frame_idx'] for j in range(start_idx, end_idx + 1)]
            ] if self.use_frame_indices else []

            csv_indices = [phys_cluster.pre_zero_index] + list(range(start_idx, end_idx + 1))

            cluster = CorrectionCluster(
                start_index=start_idx,
                end_index=end_idx,
                pre_zero_index=phys_cluster.pre_zero_index,
                timestamps=timestamps,
                frame_indices=frame_indices,
                original_values=[
                    self.df.iloc[j][displacement_col] for j in range(start_idx, end_idx + 1)
                ],
                csv_indices=csv_indices
            )

            # 添加物理群集資訊
            setattr(cluster, 'physical_cluster', phys_cluster)
            setattr(cluster, 'has_pre_zero', True)
            setattr(cluster, 'post_zero_index', phys_cluster.post_zero_index)

            correction_clusters.append(cluster)

        return correction_clusters

    def get_total_clusters(self) -> int:
        """獲取總群集數量"""
        return len(self.clusters)
    
    def get_cluster(self, index: int) -> CorrectionCluster:
        """獲取指定索引的群集"""
        if 0 <= index < len(self.clusters):
            return self.clusters[index]
        raise IndexError(f"群集索引 {index} 超出範圍")
    
    def calculate_displacement_from_lines(self, line1: ReferenceLine, line2: ReferenceLine) -> float:
        """
        基於兩條參考線段計算實際位移 (mm)
        
        Args:
            line1: 第一條參考線段 (群集前零點)
            line2: 第二條參考線段 (群集結束點)
            
        Returns:
            實際位移 (mm)，線段伸長為正 (向上移動)
        """
        # 計算線段Y分量的差異
        y_component_diff = line2.y_component - line1.y_component
        
        # 轉換為毫米 (scale_factor 代表10mm對應的像素數)
        displacement_mm = (y_component_diff * 10.0) / self.scale_factor
        
        return displacement_mm
    
    def calculate_displacement(self, point1: ReferencePoint, point2: ReferencePoint) -> float:
        """
        計算兩個參考點之間的實際位移 (mm) - 保持向後兼容
        
        Args:
            point1: 第一個參考點 (群集前零點)
            point2: 第二個參考點 (群集結束點)
            
        Returns:
            實際位移 (mm)，向上為正
        """
        # 計算Y軸像素差值 (注意：影像座標系Y軸向下為正)
        pixel_diff_y = point1.pixel_coords[1] - point2.pixel_coords[1]  # 向上為正
        
        # 轉換為毫米 (scale_factor 代表10mm對應的像素數)
        displacement_mm = (pixel_diff_y * 10.0) / self.scale_factor
        
        return displacement_mm
    
    def apply_correction(self, cluster_index: int, measured_displacement: float) -> bool:
        """
        應用校正到指定群集

        Args:
            cluster_index: 群集索引
            measured_displacement: 測量的實際位移 (mm)

        Returns:
            是否應用了校正 (如果位移太小視為雜訊則返回 False)
        """
        cluster = self.clusters[cluster_index]

        # 如果是物理群集系統，使用物理群集校正邏輯
        if self.has_frame_path and hasattr(cluster, 'physical_cluster'):
            return self.apply_physical_cluster_correction(cluster.physical_cluster, measured_displacement)

        # 舊版群集校正邏輯
        displacement_col = self.df.columns[1]
        
        # 計算最小位移閾值 (基於比例尺的10%)
        min_displacement_threshold = (10.0 / self.scale_factor) * 0.1  # 0.1像素對應的mm
        
        # 如果測量位移小於閾值，視為雜訊
        if abs(measured_displacement) < min_displacement_threshold:
            print(f"位移 {measured_displacement:.3f}mm 小於閾值 {min_displacement_threshold:.3f}mm，視為雜訊")
            
            # 將整個群集設為零
            for idx in range(cluster.start_index, cluster.end_index + 1):
                self.df.iloc[idx, 1] = 0.0
            
            return False
        
        # 計算原始值的總和 (絕對值)
        total_original = sum(abs(val) for val in cluster.original_values)
        
        if total_original == 0:
            return False
        
        # 按比例分配校正值
        for i, original_val in enumerate(cluster.original_values):
            csv_idx = cluster.start_index + i
            
            if original_val == 0:
                corrected_val = 0
            else:
                # 按原始值的比例分配測量位移
                ratio = abs(original_val) / total_original
                corrected_val = measured_displacement * ratio
                
                # 保持原始正負號
                if original_val < 0:
                    corrected_val = -corrected_val
            
            self.df.iloc[csv_idx, 1] = corrected_val
        
        return True

    def apply_physical_cluster_correction(self, physical_cluster: PhysicalCluster, measured_displacement: float) -> bool:
        """對整個物理群集區間應用校正"""
        displacement_col = self.df.columns[2]  # frame_idx, second, displacement, frame_path

        # 計算最小位移閾值
        min_displacement_threshold = (10.0 / self.scale_factor) * 0.1

        # 如果測量位移小於閾值，視為雜訊
        if abs(measured_displacement) < min_displacement_threshold:
            print(f"位移 {measured_displacement:.3f}mm 小於閾值 {min_displacement_threshold:.3f}mm，視為雜訊")

            # 將整個物理群集區間設為零
            for i in range(physical_cluster.pre_zero_index, physical_cluster.post_zero_index + 1):
                self.df.iloc[i, 2] = 0.0

            return False

        # 獲取區間內所有非零值的位置和值
        region_start = physical_cluster.pre_zero_index
        region_end = physical_cluster.post_zero_index

        non_zero_indices = []
        non_zero_values = []

        for i in range(region_start, region_end + 1):
            value = self.df.iloc[i, 2]  # displacement column
            if value != 0:
                non_zero_indices.append(i)
                non_zero_values.append(value)

        if not non_zero_values:
            print("⚠️  警告：物理群集區間內無非零值")
            return False

        # 按比例分配校正值
        total_original = sum(abs(val) for val in non_zero_values)
        if total_original == 0:
            return False

        for idx, original_val in zip(non_zero_indices, non_zero_values):
            ratio = abs(original_val) / total_original
            corrected_val = measured_displacement * ratio

            # 保持原始正負號
            if original_val < 0:
                corrected_val = -corrected_val

            self.df.iloc[idx, 2] = corrected_val

        print(f"✅ 物理群集 {physical_cluster.cluster_id} 校正完成：{len(non_zero_indices)} 個點")
        return True

    def save_corrected_csv(self) -> str:
        """
        儲存校正後的CSV檔案
        
        Returns:
            儲存的檔案路徑
        """
        # 生成新的檔名 (添加 m 前綴)
        original_path = Path(self.csv_path)
        new_filename = f"m{original_path.name}"
        new_path = original_path.parent / new_filename
        
        # 儲存檔案
        self.df.to_csv(new_path, index=False)
        
        return str(new_path)

class VideoHandler:
    """影片處理模組"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.video_name = Path(video_path).name
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"無法開啟影片檔案: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.rotation_angle = rotation_config.get(self.video_name, 0)
        
        print(f"影片載入成功: {self.video_name}")
        print(f"FPS: {self.fps}, 總幀數: {self.total_frames}")
        if self.rotation_angle != 0:
            print(f"旋轉角度: {self.rotation_angle}°")

    def get_frame_at_index(self, frame_number) -> Optional[np.ndarray]:
        """獲取指定幀號的影片幀"""
        # 確保幀號是整數
        frame_number = int(frame_number)
        
        print(f"\n=== 精確幀號提取 ===")
        print(f"目標幀號: {frame_number} (整數轉換)")
        print(f"影片FPS: {self.fps:.3f}")
        print(f"總幀數: {self.total_frames}")
        print(f"對應時戳: {frame_number / self.fps:.6f}s")
        
        if frame_number >= self.total_frames:
            print(f"❌ 錯誤: 幀號 {frame_number} 超出範圍 (總幀數: {self.total_frames})")
            return None
        
        if frame_number < 0:
            print(f"❌ 錯誤: 幀號 {frame_number} 不能為負數")
            return None
        
        success = False
        max_attempts = 3
        for attempt in range(max_attempts):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            position_error = abs(actual_pos - frame_number)
            print(f"嘗試 {attempt+1}: 設置幀位置 目標={frame_number}, 實際={int(actual_pos)}, 誤差={position_error:.1f}")
            
            if position_error < 1:
                ret, frame = self.cap.read()
                if ret:
                    print(f"✅ 成功讀取幀 {frame_number} (嘗試 {attempt+1})")
                    success = True
                    break
                else:
                    print(f"❌ 幀位置正確但讀取失敗 (嘗試 {attempt+1})")
            else:
                print(f"⚠️ 幀位置誤差過大 (嘗試 {attempt+1})")
            
            print(f"🔄 重置 VideoCapture (嘗試 {attempt+1})")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not success:
            print("🔄 使用全新 VideoCapture 最後重試...")
            temp_cap = cv2.VideoCapture(self.video_path)
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            actual_final = temp_cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(f"最終幀位置: 目標={frame_number}, 實際={int(actual_final)}")
            ret, frame = temp_cap.read()
            temp_cap.release()
            if not ret:
                print(f"❌ 最終錯誤: 無法讀取幀 {frame_number}")
                return None
            else:
                print(f"✅ 最終成功讀取幀 {frame_number}")
        
        if self.rotation_angle != 0:
            print(f"🔄 應用旋轉校正: {self.rotation_angle}°")
            frame = rotate_frame(frame, self.rotation_angle)
        
        print(f"✅ 幀提取完成 - 幀號: {frame_number}, 尺寸: {frame.shape}")
        print(f"📊 驗證: 計算時戳 = {frame_number / self.fps:.6f}s")
        print("===================\n")
        return frame

    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """獲取指定時戳的影片幀 (舊版，可能有偏差)"""
        frame_number = int(timestamp * self.fps)
        print(f"\n=== 時戳估算提取 ===")
        print(f"⚠️ 警告: 使用時戳估算，精度可能不如直接幀號")
        print(f"輸入時戳: {timestamp:.6f}s")
        print(f"影片FPS: {self.fps:.3f}")
        print(f"估算幀號: {frame_number}")
        print(f"估算誤差: ±{0.5/self.fps:.6f}s")
        print("=====================")
        return self.get_frame_at_index(frame_number)

    def load_jpg_frame(self, jpg_filename: str) -> Optional[np.ndarray]:
        """載入匯出的JPG檔案作為參考幀"""
        video_name = os.path.splitext(self.video_name)[0]
        jpg_path = os.path.join('lifts', 'exported_frames', video_name, jpg_filename)

        if not os.path.exists(jpg_path):
            print(f"❌ JPG檔案不存在: {jpg_path}")
            return None

        frame = cv2.imread(jpg_path)
        if frame is None:
            print(f"❌ 無法載入JPG檔案: {jpg_path}")
            return None

        # 應用旋轉（如果有設定）
        if self.rotation_angle != 0:
            frame = rotate_frame(frame, self.rotation_angle)

        print(f"✅ 成功載入JPG: {jpg_filename}")
        return frame

    def __del__(self):
        """清理資源"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

class CorrectionApp:
    """半自動校正GUI應用程式"""
    
    def __init__(self, root: tk.Tk, data_manager: DataManager, video_handler: VideoHandler,
                 map_frames_enabled: bool = False, map_intercept: float = 318.0, map_slope: float = 0.9946):
        self.root = root
        self.data_manager = data_manager
        self.video_handler = video_handler
        self.map_frames_enabled = map_frames_enabled
        self.map_intercept = map_intercept
        self.map_slope = map_slope
        
        # 校正狀態
        self.current_cluster_index = 0
        self.current_phase = "roi_selection"  # roi_selection, line_marking_1, line_marking_2
        self.current_line_index = 0  # 0: 第一條線段, 1: 第二條線段
        self.current_point_in_line = 0  # 0: 線段起點, 1: 線段終點
        self.reference_lines = []  # 儲存當前群集的參考線段
        self.current_line_points = []  # 儲存當前正在標記的線段點 [(x1,y1), (x2,y2)]
        self.roi_rect = None  # (x, y, width, height)
        self.zoom_factor = 8  # 增加到8倍放大以提高精度

        # 參考線段顯示控制
        self.show_reference_lines = True  # H鍵可切換

        # 重複標註功能
        self.line_annotations = [[], []]  # 每條線段的多次標註記錄 [line1_annotations, line2_annotations]
        self.max_annotations = 3  # 最多保留3次標註
        
        # GUI 組件
        self.setup_ui()
        
        # 鍵盤綁定
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
    def setup_ui(self):
        """設置使用者界面"""
        self.root.deiconify()
        # 初始標題（會在 show_current_cluster 中更新）
        self.root.title("半自動位移校正工具 - 載入中...")
        self.root.geometry("1200x800")
        
        # 頂部資訊欄
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.info_label.pack(side=tk.LEFT)
        
        # 主畫布
        self.canvas = tk.Canvas(self.root, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 綁定滑鼠事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # 底部狀態欄
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT)
        
        self.help_label = ttk.Label(status_frame, text="快捷鍵: [N]ext [B]ack [S]ave [Q]uit [H]ide線段 [R]epeat [Z]取消", font=("Arial", 9))
        self.help_label.pack(side=tk.RIGHT)
        
        # 滑鼠事件變數
        self.drag_start = None
        self.current_rect_id = None
        
    def start_correction(self):
        """開始校正流程"""
        if self.data_manager.get_total_clusters() == 0:
            messagebox.showinfo("完成", "沒有需要校正的群集")
            return
        
        self.show_current_cluster()
        
    def show_current_cluster(self):
        """顯示當前群集的標記點"""
        cluster = self.data_manager.get_cluster(self.current_cluster_index)

        # 只在開始新群集時重置狀態
        if self.current_phase == "roi_selection":
            # 新群集開始，重置所有狀態
            self.reference_lines = []
            self.current_line_points = []
            self.roi_rect = None
            self.current_line_index = 0
            self.current_point_in_line = 0
            # 重置標註記錄
            self.line_annotations = [[], []]
        
        # 檢查是否有前零點
        has_pre_zero = getattr(cluster, 'has_pre_zero', True)
        
        # 決定要顯示的時戳和幀號
        if self.current_phase in ["roi_selection", "line_marking_1"]:
            if has_pre_zero:
                timestamp = cluster.timestamps[0]  # 前零點
                frame_id = int(cluster.frame_indices[0]) if cluster.frame_indices else None
                description = "群集前零點 (第一條線段)"
            else:
                timestamp = cluster.timestamps[0]  # 群集開始點（第一行就有位移）
                frame_id = int(cluster.frame_indices[0]) if cluster.frame_indices else None
                description = "群集開始點 (檔案開頭)"
        else:  # line_marking_2
            timestamp = cluster.timestamps[-1]  # 群集結束點
            frame_id = int(cluster.frame_indices[-1]) if cluster.frame_indices else None
            description = "群集結束點 (第二條線段)"
            
        # 添加調試信息（包含幀號）
        print(f"\n=== 時戳/幀號調試信息 ===")
        print(f"當前階段: {self.current_phase}")
        print(f"群集索引: {cluster.start_index} 到 {cluster.end_index}")
        print(f"有前零點: {has_pre_zero}")
        print(f"時戳數組: {cluster.timestamps}")
        # 將幀號轉換為整數顯示
        frame_indices_int = [int(f) for f in cluster.frame_indices] if cluster.frame_indices else []
        print(f"幀號數組: {frame_indices_int}")
        print(f"選中時戳: {timestamp:.6f}s (索引: {'0' if self.current_phase in ['roi_selection', 'line_marking_1'] else '-1'})")
        print(f"選中幀號: {frame_id} (索引: {'0' if self.current_phase in ['roi_selection', 'line_marking_1'] else '-1'})")
        mapped_frame_id = None
        if self.map_frames_enabled and frame_id is not None and self.data_manager.use_frame_indices:
            mapped_frame_id = map_frame_index(frame_id, self.map_intercept, self.map_slope)
            print(f"➡️  映射後幀號: {mapped_frame_id}  (公式: {self.map_intercept} + {self.map_slope} × {frame_id})")
        print(f"時戳差異: {cluster.timestamps[-1] - cluster.timestamps[0]:.6f}s")
        if cluster.frame_indices and len(cluster.frame_indices) > 1:
            print(f"幀號差異: {int(cluster.frame_indices[-1]) - int(cluster.frame_indices[0])} 幀")
        if len(cluster.original_values) > 0:
            print(f"原始位移值: {cluster.original_values}")
            print(f"位移總和: {sum(abs(v) for v in cluster.original_values):.3f}mm")
            # 計算理論像素差異來幫助用戶識別
            expected_pixel_movement = (sum(abs(v) for v in cluster.original_values) * self.data_manager.scale_factor) / 10.0
            print(f"📏 預期位移: {sum(abs(v) for v in cluster.original_values):.3f}mm ≈ {expected_pixel_movement:.1f} 像素")
            print(f"💡 提示: 在標記時請注意這個預期的像素移動量")
        print("=========================")
        
        # 更新資訊（包含幀號和物理群集資訊）
        total_clusters = self.data_manager.get_total_clusters()
        cluster_info = f"檔案: {self.video_handler.video_name} | "

        # 如果使用物理群集系統，顯示物理群集資訊
        if self.data_manager.has_frame_path and hasattr(cluster, 'physical_cluster'):
            physical_cluster = cluster.physical_cluster
            cluster_info += f"物理群集: {self.current_cluster_index + 1}/{total_clusters} | "
            cluster_info += f"ID: {physical_cluster.cluster_id} | {description}"
            cluster_info += f" | 運動點數: {len([v for v in physical_cluster.region_values if v != 0])}"
            if used_jpg:
                cluster_info += " | 使用JPG"
        else:
            cluster_info += f"群集: {self.current_cluster_index + 1}/{total_clusters} | "
            cluster_info += f"時戳: {timestamp:.3f}s"
            if frame_id is not None:
                if mapped_frame_id is not None:
                    cluster_info += f" | 幀號: {frame_id} → {mapped_frame_id}"
                else:
                    cluster_info += f" | 幀號: {frame_id}"
            cluster_info += f" | {description}"
        
        self.info_label.config(text=cluster_info)
        
        # 更新視窗標題（包含當前群集和幀號信息）
        window_title = f"半自動位移校正工具 - {self.video_handler.video_name}"
        window_title += f" | 群集 {self.current_cluster_index + 1}/{total_clusters}"
        if frame_id is not None:
            if mapped_frame_id is not None:
                window_title += f" | 幀號: {frame_id}→{mapped_frame_id}"
            else:
                window_title += f" | 幀號: {frame_id}"
        window_title += f" | 時戳: {timestamp:.3f}s"
        self.root.title(window_title)
        
        # 優先使用JPG檔案（物理群集系統）
        frame = None
        used_jpg = False

        if self.data_manager.has_frame_path and hasattr(cluster, 'physical_cluster'):
            physical_cluster = cluster.physical_cluster

            if self.current_phase in ["roi_selection", "line_marking_1"]:
                # 第一條線段：前0點
                jpg_filename = physical_cluster.pre_zero_jpg
                frame = self.video_handler.load_jpg_frame(jpg_filename)
                if frame is not None:
                    used_jpg = True
                    print(f"✅ 使用前0點JPG: {jpg_filename}")
                    description = f"物理群集 {physical_cluster.cluster_id} 前0點 (運動前狀態)"

            elif self.current_phase == "line_marking_2":
                # 第二條線段：後0點
                jpg_filename = physical_cluster.post_zero_jpg
                frame = self.video_handler.load_jpg_frame(jpg_filename)
                if frame is not None:
                    used_jpg = True
                    print(f"✅ 使用後0點JPG: {jpg_filename}")
                    description = f"物理群集 {physical_cluster.cluster_id} 後0點 (運動後狀態)"

        # 回退到影片幀載入（如果JPG不可用）
        if frame is None:
            if frame_id is not None and self.data_manager.use_frame_indices:
                target_frame_id = mapped_frame_id if mapped_frame_id is not None else frame_id
                frame = self.video_handler.get_frame_at_index(target_frame_id)
                print(f"🔄 回退使用影片幀號 {frame_id} 進行定位")
            else:
                frame = self.video_handler.get_frame_at_timestamp(timestamp)
                print(f"🔄 回退使用時戳 {timestamp:.3f}s 進行估算定位")
            
        if frame is None:
            error_msg = f"無法獲取"
            if frame_id is not None:
                error_msg += f"幀號 {frame_id} (時戳 {timestamp:.3f}s)"
            else:
                error_msg += f"時戳 {timestamp:.3f}s"
            error_msg += " 的影片幀"
            messagebox.showerror("錯誤", error_msg)
            return
        
        self.show_frame(frame)
        
        # 更新狀態
        if self.current_phase == "roi_selection":
            if not has_pre_zero:
                self.status_label.config(text="⚠️ 故障檢測: 檔案開頭即有位移，請檢視畫面後按 [N] 選擇處理方式")
            else:
                # 計算預期位移提示
                if len(cluster.original_values) > 0:
                    expected_mm = sum(abs(v) for v in cluster.original_values)
                    expected_pixels = (expected_mm * self.data_manager.scale_factor) / 10.0
                    self.status_label.config(text=f"階段1: 請拖拽選擇ROI區域 | 預期位移: {expected_mm:.1f}mm ({expected_pixels:.1f}像素)")
                else:
                    self.status_label.config(text="階段1: 請拖拽選擇包含參考點的ROI區域")
        
    def show_frame(self, frame: np.ndarray):
        """在畫布上顯示影片幀"""
        # 轉換顏色格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 獲取畫布尺寸
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 計算縮放比例以適應畫布
        h, w = frame_rgb.shape[:2]
        scale_x = canvas_width / w
        scale_y = canvas_height / h
        self.display_scale = min(scale_x, scale_y, 1.0)  # 不放大，只縮小
        
        # 調整影像大小
        new_width = int(w * self.display_scale)
        new_height = int(h * self.display_scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # 轉換為 PIL 圖像然後為 PhotoImage
        from PIL import Image, ImageTk
        pil_image = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清除畫布並顯示圖像
        self.canvas.delete("all")
        
        # 計算置中位置
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        # 儲存圖像在畫布中的位置和尺寸
        self.image_bounds = (x_offset, y_offset, new_width, new_height)
        self.original_frame = frame
        print(f"[DEBUG] 更新 original_frame，尺寸: {frame.shape}，第一個像素: {frame[0,0]}")
        
    def on_canvas_click(self, event):
        """滑鼠點擊事件"""
        if self.current_phase == "roi_selection":
            self.drag_start = (event.x, event.y)
            
        elif self.current_phase in ["line_marking_1", "line_marking_2"]:
            # 線段標記模式
            self.place_line_point(event.x, event.y)
    
    def on_canvas_drag(self, event):
        """滑鼠拖拽事件"""
        if self.current_phase == "roi_selection" and self.drag_start:
            # 移除之前的矩形
            if self.current_rect_id:
                self.canvas.delete(self.current_rect_id)
            
            # 繪製新的選擇矩形
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            
            # 確保矩形有正確的方向
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            self.current_rect_id = self.canvas.create_rectangle(
                left, top, right, bottom,
                outline="red", width=2, dash=(5, 5)
            )
    
    def on_canvas_release(self, event):
        """滑鼠釋放事件"""
        if self.current_phase == "roi_selection" and self.drag_start:
            # 完成ROI選擇
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            
            # 計算ROI矩形
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            # 檢查ROI大小
            if (right - left) < 50 or (bottom - top) < 50:
                messagebox.showwarning("警告", "ROI區域太小，請重新選擇")
                self.canvas.delete(self.current_rect_id)
                self.current_rect_id = None
                self.drag_start = None
                return
            
            # 轉換畫布座標到原始影像座標
            img_x, img_y, img_w, img_h = self.image_bounds
            
            # 確保ROI在圖像範圍內
            left = max(img_x, left)
            top = max(img_y, top)
            right = min(img_x + img_w, right)
            bottom = min(img_y + img_h, bottom)
            
            # 轉換為原始影像座標
            roi_x = int((left - img_x) / self.display_scale)
            roi_y = int((top - img_y) / self.display_scale)
            roi_w = int((right - left) / self.display_scale)
            roi_h = int((bottom - top) / self.display_scale)
            
            self.roi_rect = (roi_x, roi_y, roi_w, roi_h)
            
            # 顯示ROI已選擇的提示
            self.status_label.config(text="ROI已選擇，按 [N] 進入線段標記模式")
            
            self.drag_start = None
    
    def enter_precision_marking_mode(self):
        """進入精細標記模式"""
        # 注意：不要在這裡改變 current_phase，它已經在調用者中設置了
        
        # 提取ROI並放大
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        print(f"[DEBUG] 提取ROI: original_frame尺寸={self.original_frame.shape}, ROI=({roi_x},{roi_y},{roi_w},{roi_h})")
        print(f"[DEBUG] ROI區域第一個像素: {self.original_frame[roi_y,roi_x] if roi_y < self.original_frame.shape[0] and roi_x < self.original_frame.shape[1] else 'out of bounds'}")
        roi_frame = self.original_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 放大到8倍
        enlarged_roi = cv2.resize(roi_frame, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        # 只顯示放大的ROI，不要更新 original_frame
        self.display_frame_only(enlarged_roi)
        
        # 顯示已標記的線段（如果有）
        self.redraw_existing_lines()
    
    def redraw_existing_lines(self):
        """重新繪製已標記的線段"""
        # 清除之前的參考線段
        self.canvas.delete("existing_line")

        # 只有在顯示模式開啟時才繪製
        if not self.show_reference_lines:
            return

        for i, line in enumerate(self.reference_lines):
            start_canvas_coords = self.pixel_to_canvas_coords(line.start_pixel_coords)
            end_canvas_coords = self.pixel_to_canvas_coords(line.end_pixel_coords)

            if start_canvas_coords and end_canvas_coords:
                # 使用不同顏色區分第一條和第二條線段，降低線寬
                color = "cyan" if i == 0 else "yellow"
                line_width = 2  # 從4降低到2
                point_size = 3  # 從6降低到3

                # 繪製線段
                self.canvas.create_line(
                    start_canvas_coords[0], start_canvas_coords[1],
                    end_canvas_coords[0], end_canvas_coords[1],
                    fill=color, width=line_width, tags="existing_line"
                )

                # 繪製端點（縮小尺寸）
                self.canvas.create_oval(
                    start_canvas_coords[0] - point_size, start_canvas_coords[1] - point_size,
                    start_canvas_coords[0] + point_size, start_canvas_coords[1] + point_size,
                    fill=color, outline="white", width=1, tags="existing_line"
                )
                self.canvas.create_oval(
                    end_canvas_coords[0] - point_size, end_canvas_coords[1] - point_size,
                    end_canvas_coords[0] + point_size, end_canvas_coords[1] + point_size,
                    fill=color, outline="white", width=1, tags="existing_line"
                )
    
    def display_frame_only(self, frame: np.ndarray):
        """只顯示幀而不更新 original_frame"""
        # 轉換顏色格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 獲取畫布尺寸
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 計算縮放比例以適應畫布
        h, w = frame_rgb.shape[:2]
        scale_x = canvas_width / w
        scale_y = canvas_height / h
        self.display_scale = min(scale_x, scale_y, 1.0)  # 不放大，只縮小
        
        # 調整影像大小
        new_width = int(w * self.display_scale)
        new_height = int(h * self.display_scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # 轉換為 PIL 圖像然後為 PhotoImage
        from PIL import Image, ImageTk
        pil_image = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清除畫布並顯示圖像
        self.canvas.delete("all")
        
        # 計算置中位置
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        
        self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.photo)
        
        # 更新圖像邊界但不更新 original_frame
        self.image_bounds = (x_offset, y_offset, new_width, new_height)
        print(f"[DEBUG] 只顯示幀，尺寸: {frame.shape}，不更新 original_frame")
    
    def place_line_point(self, canvas_x: int, canvas_y: int):
        """放置線段端點標記"""
        # 轉換畫布座標到放大後ROI的座標
        img_x, img_y, img_w, img_h = self.image_bounds
        
        if (canvas_x < img_x or canvas_x > img_x + img_w or
            canvas_y < img_y or canvas_y > img_y + img_h):
            return  # 點擊在圖像外
        
        # 轉換為放大後ROI中的座標
        roi_local_x = int((canvas_x - img_x) / self.display_scale)
        roi_local_y = int((canvas_y - img_y) / self.display_scale)
        
        # 轉換回原始影像座標
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        original_x = roi_x + (roi_local_x // self.zoom_factor)
        original_y = roi_y + (roi_local_y // self.zoom_factor)
        
        print(f"[DEBUG] 線段點座標轉換:")
        print(f"  畫布點擊: ({canvas_x}, {canvas_y})")
        print(f"  圖像邊界: {self.image_bounds}")
        print(f"  ROI本地: ({roi_local_x}, {roi_local_y})")
        print(f"  縮放因子: {self.zoom_factor}, 顯示縮放: {self.display_scale}")
        print(f"  縮放調整: ({roi_local_x // self.zoom_factor}, {roi_local_y // self.zoom_factor})")
        print(f"  ROI範圍: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
        print(f"  最終座標: ({original_x}, {original_y})")
        
        # 儲存點座標
        roi_coords = (roi_local_x // self.zoom_factor, roi_local_y // self.zoom_factor)
        pixel_coords = (original_x, original_y)
        
        if self.current_point_in_line == 0:
            # 第一個點 - 清除當前標記（保留已完成的線段）
            self.canvas.delete("line_marker")
            self.current_line_points = [pixel_coords]
            
            # 繪製起點標記
            self.draw_point_marker(canvas_x, canvas_y, "line_start")
            
            self.current_point_in_line = 1
            self.update_status_message()
            
        else:
            # 第二個點 - 完成線段
            self.current_line_points.append(pixel_coords)
            
            # 繪製終點標記
            self.draw_point_marker(canvas_x, canvas_y, "line_end")
            
            # 繪製連接線（更粗的線寬以便觀察）
            start_canvas_coords = self.pixel_to_canvas_coords(self.current_line_points[0])
            end_canvas_coords = self.pixel_to_canvas_coords(self.current_line_points[1])
            
            if start_canvas_coords and end_canvas_coords:
                self.canvas.create_line(
                    start_canvas_coords[0], start_canvas_coords[1],
                    end_canvas_coords[0], end_canvas_coords[1],
                    fill="lime", width=6, tags="line_marker"  # 增加線寬
                )
            
            # 儲存完整的線段標註
            cluster = self.data_manager.get_cluster(self.current_cluster_index)

            if self.current_line_index == 0:
                timestamp = cluster.timestamps[0]
                csv_index = cluster.csv_indices[0]
            else:
                timestamp = cluster.timestamps[-1]
                csv_index = cluster.csv_indices[-1]

            line = ReferenceLine(
                timestamp=timestamp,
                start_pixel_coords=self.current_line_points[0],
                end_pixel_coords=self.current_line_points[1],
                csv_index=csv_index,
                start_roi_coords=(0, 0),  # 簡化：這裡主要記錄像素座標
                end_roi_coords=roi_coords
            )

            # 將標註添加到記錄中（支援多次標註）
            self.add_line_annotation(line)

            # 重置線段標記狀態
            self.current_point_in_line = 0
            self.current_line_points = []

            self.update_status_message()
    
    def update_status_message(self):
        """更新狀態提示訊息"""
        base_message = ""

        if self.current_phase == "roi_selection":
            base_message = "階段1: 請拖拽選擇 ROI 區域，完成後按 [N] 確認"
        elif self.current_phase == "line_marking_1":
            line1_count = len(self.line_annotations[0])
            if self.current_point_in_line == 0:
                base_message = f"階段2a: 8倍放大精細標記 - 請點擊第一條參考線段的起點 [已標註: {line1_count}/3]"
            else:
                base_message = f"階段2b: 請點擊第一條參考線段的終點 [已標註: {line1_count}/3]"
        elif self.current_phase == "line_marking_2":
            line1_count = len(self.line_annotations[0])
            line2_count = len(self.line_annotations[1])
            if self.current_point_in_line == 0:
                base_message = f"階段3a: 8倍放大對比標記 - 青色線為第一條線段[{line1_count}/3]，請標記第二條線段起點 [{line2_count}/3]"
            else:
                base_message = f"階段3b: 請點擊第二條線段終點 [{line2_count}/3]"

        # 添加參考線段狀態
        reference_status = "顯示" if self.show_reference_lines else "隱藏"
        final_message = f"{base_message} | 參考線段: {reference_status}"

        self.status_label.config(text=final_message)
            
    def draw_point_marker(self, canvas_x: int, canvas_y: int, marker_type: str):
        """繪製點標記"""
        size = 8
        color = "lime" if marker_type == "line_start" else "orange"
        
        # 繪製小圓點
        self.canvas.create_oval(
            canvas_x - size, canvas_y - size,
            canvas_x + size, canvas_y + size,
            fill=color, outline="white", width=2, tags="line_marker"
        )
        
        # 繪製小十字
        cross_size = 4
        self.canvas.create_line(
            canvas_x - cross_size, canvas_y,
            canvas_x + cross_size, canvas_y,
            fill="white", width=2, tags="line_marker"
        )
        self.canvas.create_line(
            canvas_x, canvas_y - cross_size,
            canvas_x, canvas_y + cross_size,
            fill="white", width=2, tags="line_marker"
        )
    
    def pixel_to_canvas_coords(self, pixel_coords: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """將原始影像像素座標轉換為畫布座標"""
        if not self.roi_rect or not hasattr(self, 'image_bounds'):
            return None
            
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        img_x, img_y, img_w, img_h = self.image_bounds
        
        # 轉換為ROI本地座標
        local_x = pixel_coords[0] - roi_x
        local_y = pixel_coords[1] - roi_y
        
        # 檢查是否在ROI範圍內
        if local_x < 0 or local_x >= roi_w or local_y < 0 or local_y >= roi_h:
            return None
        
        # 轉換為畫布座標
        canvas_x = img_x + (local_x * self.zoom_factor * self.display_scale)
        canvas_y = img_y + (local_y * self.zoom_factor * self.display_scale)
        
        return (int(canvas_x), int(canvas_y))
    
    def place_reference_point(self, canvas_x: int, canvas_y: int):
        """放置參考點標記"""
        # 轉換畫布座標到放大後ROI的座標
        img_x, img_y, img_w, img_h = self.image_bounds
        
        if (canvas_x < img_x or canvas_x > img_x + img_w or
            canvas_y < img_y or canvas_y > img_y + img_h):
            return  # 點擊在圖像外
        
        # 轉換為放大後ROI中的座標
        roi_local_x = int((canvas_x - img_x) / self.display_scale)
        roi_local_y = int((canvas_y - img_y) / self.display_scale)
        
        # 轉換回原始影像座標
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        original_x = roi_x + (roi_local_x // self.zoom_factor)
        original_y = roi_y + (roi_local_y // self.zoom_factor)
        
        # 移除之前的標記
        self.canvas.delete("crosshair")
        
        # 繪製十字線 (4像素寬，相當於原影像1像素)
        crosshair_size = 20
        line_width = 4
        
        # 垂直線
        self.canvas.create_line(
            canvas_x, canvas_y - crosshair_size,
            canvas_x, canvas_y + crosshair_size,
            fill="lime", width=line_width, tags="crosshair"
        )
        
        # 水平線
        self.canvas.create_line(
            canvas_x - crosshair_size, canvas_y,
            canvas_x + crosshair_size, canvas_y,
            fill="lime", width=line_width, tags="crosshair"
        )
        
        # 儲存參考點 (如果已有點則替換)
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
        has_pre_zero = getattr(cluster, 'has_pre_zero', True)
        
        # 默認使用第一個時戳點（向後兼容舊代碼）
        timestamp = cluster.timestamps[0]
        csv_index = cluster.csv_indices[0]
        
        reference_point = ReferencePoint(
            timestamp=timestamp,
            pixel_coords=(original_x, original_y),
            csv_index=csv_index,
            roi_coords=(roi_local_x, roi_local_y)
        )
        
        # 向後兼容：如果沒有 reference_points 列表，創建一個
        if not hasattr(self, 'reference_points'):
            self.reference_points = []
        
        # 添加參考點
        self.reference_points.append(reference_point)
        
        print(f"標記參考點: 時戳={timestamp:.3f}s, 座標=({original_x}, {original_y})")
    
    def on_key_press(self, event):
        """鍵盤事件處理"""
        key = event.keysym.lower()
        
        if key == 'n':  # Next
            self.next_step()
        elif key == 'b':  # Back
            self.previous_step()
        elif key == 's':  # Save
            self.save_corrections()
        elif key == 'q':  # Quit
            self.quit_application()
        elif key == 'h':  # Hide/Show reference lines
            self.toggle_reference_lines()
        elif key == 'r':  # Repeat annotation
            self.repeat_annotation()
        elif key == 'z':  # Cancel last annotation
            self.cancel_last_annotation()
    
    def next_step(self):
        """進入下一步"""
        if self.current_phase == "roi_selection":
            # ROI選擇完成，進入第一條線段標記
            if not self.roi_rect or min(self.roi_rect[2:]) < 50:
                messagebox.showwarning("警告", "請先選擇一個有效的 ROI 區域 (最小 50x50 像素)")
                return
            
            self.current_phase = "line_marking_1"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.enter_precision_marking_mode()
            self.update_status_message()
            
        elif self.current_phase == "line_marking_1":
            # 檢查第一條線段是否完成
            if self.current_point_in_line != 0:
                messagebox.showwarning("警告", "請先完成當前線段的標記")
                return

            line1_count = len(self.line_annotations[0])
            if line1_count == 0:
                messagebox.showwarning("警告", "請先標註第一條線段")
                return
            elif line1_count < 3:
                result = messagebox.askyesno(
                    "標註數量不足",
                    f"第一條線段只有 {line1_count} 次標註（建議 3 次）\n\n"
                    f"是否繼續到第二條線段？\n\n"
                    f"點擊「否」可使用 [R] 鍵繼續標註。"
                )
                if not result:
                    return
            
            cluster = self.data_manager.get_cluster(self.current_cluster_index)
            has_pre_zero = getattr(cluster, 'has_pre_zero', True)
            
            if has_pre_zero:
                # 有前零點，移動到第二條線段（群集結束點）
                self.current_phase = "line_marking_2"
                self.current_line_index = 1
                self.current_point_in_line = 0
                # 清除當前標記，保留已完成的線段
                self.canvas.delete("line_marker")
                self.show_current_cluster()
                # 進入放大模式標記第二條線段，但保持 original_frame
                self.enter_precision_marking_mode()
                self.update_status_message()
            else:
                # 沒有前零點，可能是故障，提供清零選項
                self.handle_first_line_displacement()
                return
                
        elif self.current_phase == "line_marking_2":
            # 檢查第二條線段是否完成
            if self.current_point_in_line != 0:
                messagebox.showwarning("警告", "請先完成當前線段的標記")
                return

            line2_count = len(self.line_annotations[1])
            if line2_count == 0:
                messagebox.showwarning("警告", "請先標註第二條線段")
                return
            elif line2_count < 3:
                result = messagebox.askyesno(
                    "標註數量不足",
                    f"第二條線段只有 {line2_count} 次標註（建議 3 次）\n\n"
                    f"是否繼續計算位移？\n\n"
                    f"點擊「否」可使用 [R] 鍵繼續標註。"
                )
                if not result:
                    return
            
            # 兩條線段都已標記，計算並應用校正
            self.apply_cluster_correction()
            
            # 移動到下一個群集
            self.move_to_next_cluster()
    
    def previous_step(self):
        """返回上一步"""
        if self.current_phase == "line_marking_2":
            # 從第二條線段回到第一條線段
            self.current_phase = "line_marking_1"
            self.current_line_index = 0
            self.current_point_in_line = 0
            # 清空第二條線段的標註記錄
            self.line_annotations[1] = []
            self.update_reference_lines_from_annotations()
            self.show_current_cluster()
            self.enter_precision_marking_mode()
            self.update_status_message()
        elif self.current_phase == "line_marking_1":
            # 從第一條線段回到ROI選擇
            self.current_phase = "roi_selection"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.reference_lines = []
            self.line_annotations = [[], []]  # 清空所有標註記錄
            self.roi_rect = None
            self.show_current_cluster()
        elif self.current_cluster_index > 0:
            # 回到上一個群集
            self.current_cluster_index -= 1
            self.current_phase = "roi_selection"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.reference_lines = []
            self.line_annotations = [[], []]  # 清空所有標註記錄
            self.roi_rect = None
            self.show_current_cluster()
    
    def handle_first_line_displacement(self):
        """處理第一行就有位移的情況（可能是故障）"""
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
        
        # 顯示故障檢測對話框
        result = messagebox.askyesnocancel(
            "檢測到可能的設備故障",
            f"此群集從檔案第一行就開始有位移，這通常表示設備故障或檢測異常。\n\n"
            f"群集範圍: 第 {cluster.start_index + 1} 行到第 {cluster.end_index + 1} 行\n"
            f"位移值數量: {len(cluster.original_values)} 個\n"
            f"範例值: {cluster.original_values[:3]}...\n\n"
            f"請選擇處理方式:\n"
            f"• 是(Y): 將此群集清零（視為故障）\n"
            f"• 否(N): 保持原值並跳過校正\n"
            f"• 取消: 返回檢視"
        )
        
        if result is True:  # Yes - 清零
            self.clear_cluster_to_zero()
        elif result is False:  # No - 跳過
            self.skip_current_cluster()
        # else: Cancel - 什麼都不做，讓用戶繼續檢視
    
    def clear_cluster_to_zero(self):
        """將當前群集清零"""
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
        
        # 將群集中的所有位移值設為零
        displacement_col = self.data_manager.df.columns[1]
        for idx in range(cluster.start_index, cluster.end_index + 1):
            self.data_manager.df.iloc[idx, 1] = 0.0
        
        print(f"群集 {self.current_cluster_index + 1} 已清零（故障處理）")
        
        # 移動到下一個群集
        self.move_to_next_cluster()
    
    def skip_current_cluster(self):
        """跳過當前群集不進行校正"""
        print(f"群集 {self.current_cluster_index + 1} 已跳過校正")
        
        # 移動到下一個群集
        self.move_to_next_cluster()
    
    def move_to_next_cluster(self):
        """移動到下一個群集"""
        self.current_cluster_index += 1
        if self.current_cluster_index >= self.data_manager.get_total_clusters():
            messagebox.showinfo("完成", "所有群集處理完成！")
            self.save_corrections()
            return
        
        # 重置狀態為新群集
        self.current_phase = "roi_selection"
        self.current_line_index = 0
        self.current_point_in_line = 0
        self.reference_lines = []
        self.current_line_points = []
        self.roi_rect = None
        # 重置標註記錄
        self.line_annotations = [[], []]
        
        self.show_current_cluster()

    def apply_cluster_correction(self):
        """應用當前群集的校正"""
        if len(self.reference_lines) < 2:
            messagebox.showerror("錯誤", "需要兩條參考線段才能計算位移")
            return
        
        # 計算實際位移 (基於線段Y分量差異)
        line1 = self.reference_lines[0]  # 前零點線段
        line2 = self.reference_lines[1]  # 結束點線段
        
        measured_displacement = self.data_manager.calculate_displacement_from_lines(line1, line2)

        # 顯示線段詳細資訊（包含幀號）
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
        print(f"\n=== 線段校正計算 ===")
        print(f"群集範圍: 第 {cluster.start_index + 1} 行到第 {cluster.end_index + 1} 行")
        if cluster.frame_indices:
            print(f"幀號範圍: {int(cluster.frame_indices[0])} 到 {int(cluster.frame_indices[-1])}")
        print(f"時戳範圍: {cluster.timestamps[0]:.6f}s 到 {cluster.timestamps[-1]:.6f}s")
        print(f"第一條線段 (時戳: {line1.timestamp:.6f}s):")
        print(f"  起點: {line1.start_pixel_coords}")
        print(f"  終點: {line1.end_pixel_coords}")
        print(f"  Y分量: {line1.y_component:.1f} 像素")
        print(f"第二條線段 (時戳: {line2.timestamp:.6f}s):")
        print(f"  起點: {line2.start_pixel_coords}")
        print(f"  終點: {line2.end_pixel_coords}")
        print(f"  Y分量: {line2.y_component:.1f} 像素")
        print(f"差異計算:")
        print(f"  Y分量差異: {line2.y_component:.1f} - {line1.y_component:.1f} = {line2.y_component - line1.y_component:.1f} 像素")
        print(f"  比例尺: {self.data_manager.scale_factor} 像素/10mm")
        print(f"  計算位移: ({line2.y_component - line1.y_component:.1f} × 10) / {self.data_manager.scale_factor} = {measured_displacement:.3f} mm")
        print("=====================")

        # 計算程式原始估計值
        original_displacement = sum(abs(v) for v in cluster.original_values)

        # 位移比較警示
        if abs(measured_displacement) < original_displacement * 0.95:  # 人工測量值小於程式估計值的95%
            choice = self.show_displacement_warning(measured_displacement, original_displacement)

            if choice == "use_original":
                # 使用程式估計值
                measured_displacement = original_displacement
                print(f"用戶選擇使用程式估計值: {measured_displacement:.3f}mm")
            elif choice == "re_annotate":
                # 重新標註
                print("用戶選擇重新標註")
                return  # 不應用校正，留在當前階段
            # else: choice == "use_manual" - 使用人工測量值，繼續執行

        # 應用校正
        is_applied = self.data_manager.apply_correction(self.current_cluster_index, measured_displacement)

        if is_applied:
            print(f"群集 {self.current_cluster_index + 1} 校正完成，測量位移: {measured_displacement:.3f}mm")
        else:
            print(f"群集 {self.current_cluster_index + 1} 被視為雜訊並移除")
    
    def save_corrections(self):
        """儲存校正結果"""
        try:
            saved_path = self.data_manager.save_corrected_csv()
            messagebox.showinfo("儲存成功", f"校正後的檔案已儲存至:\n{saved_path}")
        except Exception as e:
            messagebox.showerror("儲存失敗", f"無法儲存檔案: {str(e)}")
    
    def toggle_reference_lines(self):
        """切換參考線段的顯示/隱藏"""
        self.show_reference_lines = not self.show_reference_lines
        status = "顯示" if self.show_reference_lines else "隱藏"
        print(f"參考線段已{status}")

        # 重新繪製線段（或清除）
        self.redraw_existing_lines()

        # 更新狀態訊息
        self.update_status_message()

    def repeat_annotation(self):
        """重複標註當前線段"""
        if self.current_phase not in ["line_marking_1", "line_marking_2"]:
            print("只能在線段標記模式下重複標註")
            return

        if self.current_point_in_line != 0:
            print("請先完成當前線段的標記")
            return

        # 重新開始標記當前線段
        self.current_point_in_line = 0
        self.current_line_points = []

        # 清除當前標記
        self.canvas.delete("line_marker")

        print(f"開始重複標註線段 {self.current_line_index + 1}")
        self.update_status_message()

    def cancel_last_annotation(self):
        """取消最後一次標註（不納入記錄）"""
        if self.current_phase not in ["line_marking_1", "line_marking_2"]:
            print("只能在線段標記模式下取消標註")
            return

        # 如果當前線段有標註記錄，移除最後一次
        if len(self.line_annotations[self.current_line_index]) > 0:
            removed_annotation = self.line_annotations[self.current_line_index].pop()
            print(f"已取消線段 {self.current_line_index + 1} 的最後一次標註")

            # 更新 reference_lines 顯示
            self.update_reference_lines_from_annotations()

            # 重新繪製
            self.redraw_existing_lines()
        else:
            print(f"線段 {self.current_line_index + 1} 沒有可取消的標註")

        self.update_status_message()

    def add_line_annotation(self, line: ReferenceLine):
        """添加線段標註到記錄中，支援自動剔除離群的標註"""
        current_annotations = self.line_annotations[self.current_line_index]

        # 添加新標註
        current_annotations.append(line)

        # 如果超過3次，剔除離平均最遠的那個
        if len(current_annotations) > self.max_annotations:
            self.remove_outlier_annotation()

        # 更新顯示的參考線段（使用平均值）
        self.update_reference_lines_from_annotations()

        print(f"線段 {self.current_line_index + 1} 已標註 {len(current_annotations)} 次")

    def remove_outlier_annotation(self):
        """剔除離平均值最遠的標註"""
        current_annotations = self.line_annotations[self.current_line_index]

        if len(current_annotations) <= self.max_annotations:
            return

        # 計算每個標註的Y分量
        y_components = [line.y_component for line in current_annotations]

        # 計算平均值
        mean_y = sum(y_components) / len(y_components)

        # 找到離平均最遠的索引
        max_distance = 0
        outlier_index = 0

        for i, y_comp in enumerate(y_components):
            distance = abs(y_comp - mean_y)
            if distance > max_distance:
                max_distance = distance
                outlier_index = i

        # 剔除離群值
        removed_annotation = current_annotations.pop(outlier_index)
        print(f"已剔除離群標註（Y分量: {removed_annotation.y_component:.1f}，距離平均: {max_distance:.1f}）")

    def update_reference_lines_from_annotations(self):
        """從標註記錄更新參考線段顯示（使用平均值）"""
        for line_idx in range(2):  # 兩條線段
            annotations = self.line_annotations[line_idx]

            if not annotations:
                # 沒有標註，移除對應的參考線段
                if line_idx < len(self.reference_lines):
                    self.reference_lines.pop(line_idx)
                continue

            # 計算平均線段
            avg_line = self.calculate_average_line(annotations)

            # 更新或添加到 reference_lines
            if line_idx < len(self.reference_lines):
                self.reference_lines[line_idx] = avg_line
            else:
                self.reference_lines.append(avg_line)

    def calculate_average_line(self, annotations: List[ReferenceLine]) -> ReferenceLine:
        """計算多次標註的平均線段"""
        if not annotations:
            raise ValueError("沒有標註可以計算平均")

        # 計算平均座標
        avg_start_x = sum(line.start_pixel_coords[0] for line in annotations) / len(annotations)
        avg_start_y = sum(line.start_pixel_coords[1] for line in annotations) / len(annotations)
        avg_end_x = sum(line.end_pixel_coords[0] for line in annotations) / len(annotations)
        avg_end_y = sum(line.end_pixel_coords[1] for line in annotations) / len(annotations)

        # 使用第一個標註的其他屬性
        first_annotation = annotations[0]

        return ReferenceLine(
            timestamp=first_annotation.timestamp,
            start_pixel_coords=(int(avg_start_x), int(avg_start_y)),
            end_pixel_coords=(int(avg_end_x), int(avg_end_y)),
            csv_index=first_annotation.csv_index,
            start_roi_coords=first_annotation.start_roi_coords,
            end_roi_coords=first_annotation.end_roi_coords
        )

    def show_displacement_warning(self, measured_displacement: float, original_displacement: float) -> str:
        """顯示位移比較警示對話框"""
        from tkinter import simpledialog

        # 創建自定義對話框
        dialog = tk.Toplevel(self.root)
        dialog.title("位移比較警示")
        dialog.geometry("500x350")
        dialog.modal = True
        dialog.grab_set()

        # 置中顯示
        dialog.transient(self.root)
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (350 // 2)
        dialog.geometry(f"500x350+{x}+{y}")

        # 警示文字
        warning_frame = ttk.Frame(dialog)
        warning_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(warning_frame, text="⚠️ 位移測量警示", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))

        info_text = f"""人工測量值明顯小於程式估計值：

• 人工測量值： {abs(measured_displacement):.3f} mm
• 程式估計值： {original_displacement:.3f} mm
• 差異比例： {(abs(measured_displacement) / original_displacement * 100):.1f}%

這可能表示：
1. 標註精度可能不足
2. 程式估計值可能更接近真實值
3. 圖像特徵可能不明顯

請選擇處理方式："""

        info_label = ttk.Label(warning_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(pady=(0, 20))

        # 結果變數
        result = {"choice": None}

        # 按鈕框架
        button_frame = ttk.Frame(warning_frame)
        button_frame.pack(fill=tk.X)

        def on_use_original():
            result["choice"] = "use_original"
            dialog.destroy()

        def on_re_annotate():
            result["choice"] = "re_annotate"
            dialog.destroy()

        def on_use_manual():
            result["choice"] = "use_manual"
            dialog.destroy()

        ttk.Button(button_frame, text="使用程式估計值", command=on_use_original).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="重新標註", command=on_re_annotate).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="使用人工校正值", command=on_use_manual).pack(side=tk.LEFT)

        # 等待用戶選擇
        dialog.wait_window()

        return result["choice"] or "use_manual"  # 預設使用人工值

    def quit_application(self):
        """退出應用程式"""
        if messagebox.askokcancel("確認退出", "是否要退出校正工具？\n未儲存的更改將丟失。"):
            self.root.quit()

def main():
    """主函數 - 選擇檔案並啟動校正工具"""
    # 參數：可選啟用幀映射（仿射）
    parser = argparse.ArgumentParser(description="半自動位移校正工具")
    parser.add_argument("--map-frames", action="store_true", help="啟用 CSV 幀→原檔幀的仿射映射")
    parser.add_argument("--map-intercept", type=float, default=318.0, help="幀映射截距，預設 318")
    parser.add_argument("--map-slope", type=float, default=0.9946, help="幀映射斜率，預設 0.9946")
    args, _ = parser.parse_known_args()

    # 建立根視窗但隱藏
    root = tk.Tk()
    root.withdraw()
    
    try:
        # 選擇清理後的CSV檔案
        csv_path = filedialog.askopenfilename(
            title="選擇清理後的CSV檔案",
            initialdir="lifts/result",
            filetypes=[("CSV檔案", "c*.csv"), ("所有檔案", "*.*")]
        )
        
        if not csv_path:
            return
        
        # 從CSV檔名推導影片檔名
        csv_filename = Path(csv_path).name
        if csv_filename.startswith('c'):
            video_filename = csv_filename[1:]  # 移除 'c' 前綴
            video_filename = video_filename.replace('.csv', '.mp4')
        else:
            messagebox.showerror("錯誤", "請選擇以 'c' 開頭的清理後CSV檔案")
            return
        
        # 檢查對應的影片檔案
        video_path = Path("lifts/data") / video_filename
        if not video_path.exists():
            messagebox.showerror("錯誤", f"找不到對應的影片檔案: {video_path}")
            return
        
        print(f"準備處理:")
        print(f"CSV檔案: {csv_path}")
        print(f"影片檔案: {video_path}")
        
        # 初始化數據管理器
        data_manager = DataManager(csv_path, video_filename)
        
        if data_manager.get_total_clusters() == 0:
            messagebox.showinfo("完成", "此檔案沒有需要校正的位移群集")
            return
        
        print(f"發現 {data_manager.get_total_clusters()} 個需要校正的位移群集")
        
        # 初始化影片處理器
        video_handler = VideoHandler(str(video_path))
        
        # 啟動校正界面
        app = CorrectionApp(
            root,
            data_manager,
            video_handler,
            map_frames_enabled=args.map_frames,
            map_intercept=args.map_intercept,
            map_slope=args.map_slope,
        )
        app.start_correction()
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("錯誤", f"初始化失敗: {str(e)}")
    finally:
        root.destroy()

if __name__ == '__main__':
    main()
