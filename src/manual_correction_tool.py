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
    start_pixel_coords: Tuple[float, float]  # 線段起點 (x, y) 在原始影片中的座標
    end_pixel_coords: Tuple[float, float]    # 線段終點 (x, y) 在原始影片中的座標
    csv_index: int
    start_roi_coords: Tuple[float, float]    # 線段起點在ROI中的座標
    end_roi_coords: Tuple[float, float]      # 線段終點在ROI中的座標
    
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



class DataManager:
    """數據管理模組"""
    
    def __init__(self, csv_path: str, video_name: str):
        self.csv_path = csv_path
        self.video_name = video_name
        self.df = pd.read_csv(csv_path)
        self.scale_factor = scale_config.get(video_name, None)

        # 確定位移欄位名稱和索引
        self.displacement_column = self._find_displacement_column()
        self.displacement_col_index = self.df.columns.get_loc(self.displacement_column)
        
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
            raise ValueError("CSV 檔案缺少 'frame_path' 欄位。此工具僅支援新版格式。")

    def _find_displacement_column(self) -> str:
        """智能找到位移欄位"""
        # 常見的位移欄位名稱候選
        displacement_candidates = [
            'displacement',  # 英文標準名稱
            'displacement_mm',  # 帶單位的名稱
            '位移',  # 中文名稱
            '位移_mm',  # 中文帶單位
            'vertical_travel_distance (mm)',  # lift_travel_detection 輸出格式
            'v_travel_distance',  # 縮寫版本
        ]

        # 首先嘗試按名稱匹配
        for candidate in displacement_candidates:
            if candidate in self.df.columns:
                print(f"✅ 找到位移欄位: '{candidate}'")
                return candidate

        # 按欄位位置回退（兼容舊格式）
        if len(self.df.columns) >= 3:
            displacement_col = self.df.columns[2]  # 第3欄
            print(f"⚠️ 按位置使用第3欄作為位移欄位: '{displacement_col}'")
            return displacement_col

        # 如果都找不到，拋出錯誤
        available_columns = list(self.df.columns)
        raise ValueError(
            f"無法找到位移欄位。\n"
            f"可用欄位: {available_columns}\n"
            f"請確保CSV包含位移數據欄位"
        )

    def _identify_physical_clusters_from_png_tags(self) -> List[PhysicalCluster]:
        """基於PNG標籤識別物理群集 - 極其簡化的邏輯"""
        physical_clusters = []

        # 尋找所有前0點標籤
        for i, row in self.df.iterrows():
            frame_path = row.get('frame_path', '')
            
            # 跳過 NaN 值和空字符串
            if not isinstance(frame_path, str) or not frame_path:
                continue

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
                    region_values = self.df.iloc[pre_zero_index:post_zero_index+1][self.displacement_column].tolist()

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
            non_zero_indices = []

            for i in range(phys_cluster.pre_zero_index, phys_cluster.post_zero_index + 1):
                if self.df.iloc[i][self.displacement_column] != 0:
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
                    self.df.iloc[j][self.displacement_column] for j in range(start_idx, end_idx + 1)
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
    
    def calculate_displacement(self, line1: ReferenceLine, line2: ReferenceLine) -> float:
        """
        計算兩條參考線段之間的實際位移 (mm) - 保持向後兼容
        
        Args:
            line1: 第一條參考線段 (群集前零點)
            line2: 第二條參考線段 (群集結束點)
            
        Returns:
            實際位移 (mm)，向上為正
        """
        # 計算Y分量的差異
        y_component_diff = line2.y_component - line1.y_component
        
        # 轉換為毫米 (scale_factor 代表10mm對應的像素數)
        displacement_mm = (y_component_diff * 10.0) / self.scale_factor
        
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

        # 由於已強制使用物理群集系統，直接調用其校正邏輯
        return self.apply_physical_cluster_correction(cluster.physical_cluster, measured_displacement)

    def apply_physical_cluster_correction(self, physical_cluster: PhysicalCluster, measured_displacement: float) -> bool:
        """對整個物理群集區間應用校正"""
        # 計算最小位移閾值
        min_displacement_threshold = (10.0 / self.scale_factor) * 0.1

        # 如果測量位移小於閾值，視為雜訊
        if abs(measured_displacement) < min_displacement_threshold:
            print(f"位移 {measured_displacement:.3f}mm 小於閾值 {min_displacement_threshold:.3f}mm，視為雜訊")

            # 將整個物理群集區間設為零
            for i in range(physical_cluster.pre_zero_index, physical_cluster.post_zero_index + 1):
                self.df.iloc[i, self.displacement_col_index] = 0.0

            return False

        # 獲取區間內所有非零值的位置和值
        region_start = physical_cluster.pre_zero_index
        region_end = physical_cluster.post_zero_index

        non_zero_indices = []
        non_zero_values = []

        for i in range(region_start, region_end + 1):
            value = self.df.iloc[i, self.displacement_col_index]
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

            self.df.iloc[idx, self.displacement_col_index] = corrected_val

        print(f"✅ 物理群集 {physical_cluster.cluster_id} 校正完成：{len(non_zero_indices)} 個點")
        return True

    def save_corrected_csv(self) -> str:
        """
        儲存校正後的CSV檔案

        Returns:
            儲存的檔案路徑
        """
        # 生成新的檔名 (統一使用 mc 前綴)
        original_path = Path(self.csv_path)
        original_name = original_path.name

        # 移除現有前綴，取得基本檔名
        if original_name.startswith('mc'):
            base_name = original_name[2:]  # 移除 mc 前綴
        elif original_name.startswith('c'):
            base_name = original_name[1:]  # 移除 c 前綴
        else:
            base_name = original_name  # 無前綴

        new_filename = f"mc{base_name}"
        new_path = original_path.parent / new_filename
        
        # 儲存檔案
        self.df.to_csv(new_path, index=False)
        
        return str(new_path)

class JPGHandler:
    """JPG檔案處理模組"""
    
    def __init__(self, video_name: str):
        """
        初始化JPG處理器
        
        Args:
            video_name: 影片名稱（如 '1.mp4'），用於查找JPG檔案目錄
        """
        self.video_name = video_name
        self.video_base_name = os.path.splitext(video_name)[0]
        self.rotation_angle = rotation_config.get(video_name, 0)
        
        print(f"✅ JPG處理器初始化成功: {self.video_name}")
        if self.rotation_angle != 0:
            print(f"   旋轉角度: {self.rotation_angle}°")

    def load_jpg_frame(self, jpg_filename: str) -> Optional[np.ndarray]:
        """
        載入匯出的JPG檔案作為參考幀
        
        Args:
            jpg_filename: JPG檔案名稱（如 'pre_cluster_001.jpg'）
            
        Returns:
            載入的影像幀，或None如果失敗
        """
        jpg_path = os.path.join('lifts', 'exported_frames', self.video_base_name, jpg_filename)

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

class CorrectionApp:
    """半自動校正GUI應用程式"""
    
    def __init__(self, root: tk.Tk, data_manager: DataManager, jpg_handler: JPGHandler):
        self.root = root
        self.data_manager = data_manager
        self.jpg_handler = jpg_handler
        
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
        
        # 初始化變數
        used_jpg = False
        
        # 更新資訊（包含幀號和物理群集資訊）
        total_clusters = self.data_manager.get_total_clusters()
        cluster_info = f"檔案: {self.jpg_handler.video_name} | "

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
                cluster_info += f" | 幀號: {frame_id}"
            cluster_info += f" | {description}"
        
        self.info_label.config(text=cluster_info)
        
        # 更新視窗標題（包含當前群集和時戳信息）
        window_title = f"半自動位移校正工具 - {self.jpg_handler.video_name}"
        window_title += f" | 群集 {self.current_cluster_index + 1}/{total_clusters}"
        if frame_id is not None:
            window_title += f" | 幀號: {frame_id}"
        window_title += f" | 時戳: {timestamp:.3f}s"
        self.root.title(window_title)
        
        # 加載JPG檔案（物理群集系統必需）
        frame = None

        if not self.data_manager.has_frame_path or not hasattr(cluster, 'physical_cluster'):
            messagebox.showerror("錯誤", "CSV 檔案缺少 'frame_path' 欄位或物理群集資訊\n此工具僅支援包含物理群集標籤的新版CSV格式")
            return

        physical_cluster = cluster.physical_cluster

        if self.current_phase in ["roi_selection", "line_marking_1"]:
            # 第一條線段：前0點
            jpg_filename = physical_cluster.pre_zero_jpg
            frame = self.jpg_handler.load_jpg_frame(jpg_filename)
            if frame is None:
                messagebox.showerror("錯誤", f"無法加載前0點JPG檔案: {jpg_filename}\n請確保所有物理群集JPG檔案都已匯出到 lifts/exported_frames/{self.jpg_handler.video_base_name}/ 目錄")
                return
            used_jpg = True
            print(f"✅ 使用前0點JPG: {jpg_filename}")
            description = f"物理群集 {physical_cluster.cluster_id} 前0點 (運動前狀態)"

        elif self.current_phase == "line_marking_2":
            # 第二條線段：後0點
            jpg_filename = physical_cluster.post_zero_jpg
            frame = self.jpg_handler.load_jpg_frame(jpg_filename)
            if frame is None:
                messagebox.showerror("錯誤", f"無法加載後0點JPG檔案: {jpg_filename}\n請確保所有物理群集JPG檔案都已匯出到 lifts/exported_frames/{self.jpg_handler.video_base_name}/ 目錄")
                return
            used_jpg = True
            print(f"✅ 使用後0點JPG: {jpg_filename}")
            description = f"物理群集 {physical_cluster.cluster_id} 後0點 (運動後狀態)"
        
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
        roi_local_x = (canvas_x - img_x) / self.display_scale
        roi_local_y = (canvas_y - img_y) / self.display_scale
        
        # 轉換回原始影像座標
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        original_x = roi_x + (roi_local_x / self.zoom_factor)
        original_y = roi_y + (roi_local_y / self.zoom_factor)
        
        print(f"[DEBUG] 線段點座標轉換:")
        print(f"  畫布點擊: ({canvas_x}, {canvas_y})")
        print(f"  圖像邊界: {self.image_bounds}")
        print(f"  ROI本地: ({roi_local_x}, {roi_local_y})")
        print(f"  縮放因子: {self.zoom_factor}, 顯示縮放: {self.display_scale}")
        print(f"  縮放調整: ({roi_local_x / self.zoom_factor}, {roi_local_y / self.zoom_factor})")
        print(f"  ROI範圍: ({roi_x}, {roi_y}, {roi_w}, {roi_h})")
        print(f"  最終座標: ({original_x}, {original_y})")
        
        # 儲存點座標
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

            start_pixel = self.current_line_points[0]
            end_pixel = self.current_line_points[1]
            start_roi = (start_pixel[0] - roi_x, start_pixel[1] - roi_y)
            end_roi = (end_pixel[0] - roi_x, end_pixel[1] - roi_y)

            line = ReferenceLine(
                timestamp=timestamp,
                start_pixel_coords=start_pixel,
                end_pixel_coords=end_pixel,
                csv_index=csv_index,
                start_roi_coords=start_roi,
                end_roi_coords=end_roi
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

            # 執行第一條線段的離群值剔除
            if line1_count > self.max_annotations:
                print(f"\n=== 第一條線段離群值剔除 ===")
                self.remove_outlier_annotations(0)
                # 重新更新顯示
                self.update_reference_lines_from_annotations()
                self.redraw_existing_lines()
                print("===============================\n")

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

            # 執行第二條線段的離群值剔除
            if line2_count > self.max_annotations:
                print(f"\n=== 第二條線段離群值剔除 ===")
                self.remove_outlier_annotations(1)
                # 重新更新顯示
                self.update_reference_lines_from_annotations()
                self.redraw_existing_lines()
                print("===============================\n")

            # 兩條線段都已標記，計算並應用校正
            should_move_to_next = self.apply_cluster_correction()
            
            # 只有當校正已完成時才移動到下一個群集
            # 如果用戶選擇重新標註，會返回 False，不進入下一個群集
            if should_move_to_next:
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
        for idx in range(cluster.start_index, cluster.end_index + 1):
            self.data_manager.df.iloc[idx, self.data_manager.displacement_col_index] = 0.0
        
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

    def apply_cluster_correction(self) -> bool:
        """
        應用當前群集的校正
        
        Returns:
            bool: 如果校正已應用或用戶選擇使用人工值返回True，如果用戶選擇重新標注返回False
        """
        if len(self.reference_lines) < 2:
            messagebox.showerror("錯誤", "需要兩條參考線段才能計算位移")
            return False
        
        # 計算實際位移 (基於線段Y分量差異)
        line1 = self.reference_lines[0]  # 前零點線段
        line2 = self.reference_lines[1]  # 結束點線段
        
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
        measured_displacement = self.data_manager.calculate_displacement_from_lines(line1, line2)
        original_displacement = sum(abs(v) for v in cluster.original_values)
        measured_magnitude = abs(measured_displacement)
        pixel_threshold = 3.0  # 像素
        mm_threshold = (pixel_threshold * 10.0) / self.data_manager.scale_factor
        difference_mm = measured_magnitude - original_displacement
        difference_px = abs(difference_mm) * self.data_manager.scale_factor / 10.0

        # 顯示線段詳細資訊（包含幀號）
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
        print(f"程式估計值總和: {original_displacement:.3f} mm")
        print(f"人工標記絕對值: {measured_magnitude:.3f} mm")
        print(f"位移差異: {difference_mm:+.3f} mm (≈ {difference_px:.2f} 像素)")
        print(f"容許差異閾值: {mm_threshold:.3f} mm (≈ {pixel_threshold:.1f} 像素)")
        print("=====================")

        if abs(difference_mm) >= mm_threshold:
            choice = self.show_displacement_warning(
                measured_displacement=measured_displacement,
                measured_magnitude=measured_magnitude,
                original_displacement=original_displacement,
                difference_mm=difference_mm,
                difference_px=difference_px,
                mm_threshold=mm_threshold,
                pixel_threshold=pixel_threshold
            )

            if choice == "use_original":
                # 使用程式估計值
                measured_displacement = original_displacement
                print(f"用戶選擇使用程式估計值: {measured_displacement:.3f}mm")
            elif choice == "re_annotate":
                # 重新標註 - 退回到第一條線段並清空所有標註
                print("用戶選擇重新標註，返回該群集的 ROI 圈選階段")
                self.reset_to_first_line_annotation()
                return False  # 返回 False 表示不應進入下一個群集
            # else: choice == "use_manual" - 使用人工測量值，繼續執行

        # 應用校正
        is_applied = self.data_manager.apply_correction(self.current_cluster_index, measured_displacement)

        if is_applied:
            print(f"群集 {self.current_cluster_index + 1} 校正完成，測量位移: {measured_displacement:.3f}mm")
        else:
            print(f"群集 {self.current_cluster_index + 1} 被視為雜訊並移除")
        
        return True  # 返回 True 表示校正已完成，可以進入下一個群集
    
    def save_corrections(self):
        """儲存校正結果或暫存工作狀態"""
        try:
            # 檢查是否所有群集都已處理完成
            total_clusters = self.data_manager.get_total_clusters()

            if self.current_cluster_index >= total_clusters:
                # 所有群集已完成，正常儲存CSV
                saved_path = self.data_manager.save_corrected_csv()
                messagebox.showinfo("儲存成功", f"校正後的檔案已儲存至:\n{saved_path}")
            else:
                # 工作未完成，詢問用戶是否要暫存
                remaining = total_clusters - self.current_cluster_index
                result = messagebox.askyesno(
                    "工作未完成",
                    f"目前進度: {self.current_cluster_index}/{total_clusters} 群集已完成\n"
                    f"還有 {remaining} 個群集待處理\n\n"
                    f"是否要暫存目前的工作狀態？\n"
                    f"（選擇「否」將強制儲存CSV檔案）"
                )

                if result:
                    # 暫存工作狀態
                    temp_path = self.save_temporary_state()
                    messagebox.showinfo(
                        "暫存成功",
                        f"工作狀態已暫存至:\n{temp_path}\n\n"
                        f"下次開啟相同CSV檔案時可選擇載入此暫存狀態"
                    )
                else:
                    # 強制儲存CSV
                    saved_path = self.data_manager.save_corrected_csv()
                    messagebox.showinfo("強制儲存成功", f"校正後的檔案已儲存至:\n{saved_path}")

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
        """添加線段標註到記錄中，延遲到按N時才剔除離群值"""
        current_annotations = self.line_annotations[self.current_line_index]

        # 添加新標註（不限制數量）
        current_annotations.append(line)

        # 更新顯示的參考線段（使用平均值）
        self.update_reference_lines_from_annotations()

        print(f"線段 {self.current_line_index + 1} 已標註 {len(current_annotations)} 次")
        if len(current_annotations) > self.max_annotations:
            print(f"  ⚠️  超過建議數量 {self.max_annotations} 次，將在按 [N] 時自動剔除離群值")

    def remove_outlier_annotations(self, line_index: int):
        """批量剔除指定線段中離平均值最遠的標註，保留最多3個"""
        current_annotations = self.line_annotations[line_index]

        if len(current_annotations) <= self.max_annotations:
            return

        # 計算需要剔除的數量
        num_to_remove = len(current_annotations) - self.max_annotations
        print(f"線段 {line_index + 1}：需要從 {len(current_annotations)} 次標註中剔除 {num_to_remove} 個離群值")

        # 重複剔除直到達到目標數量
        for round_num in range(num_to_remove):
            if len(current_annotations) <= self.max_annotations:
                break

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
            print(f"  第 {round_num + 1} 輪剔除：Y分量 {removed_annotation.y_component:.1f}，距離平均 {max_distance:.1f}")

        print(f"✅ 線段 {line_index + 1} 剔除完成，保留 {len(current_annotations)} 次標註")

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
            start_pixel_coords=(avg_start_x, avg_start_y),
            end_pixel_coords=(avg_end_x, avg_end_y),
            csv_index=first_annotation.csv_index,
            start_roi_coords=first_annotation.start_roi_coords,
            end_roi_coords=first_annotation.end_roi_coords
        )

    def show_displacement_warning(
        self,
        measured_displacement: float,
        measured_magnitude: float,
        original_displacement: float,
        difference_mm: float,
        difference_px: float,
        mm_threshold: float,
        pixel_threshold: float
    ) -> str:
        """顯示位移比較警示對話框"""
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

        direction = "較大" if difference_mm >= 0 else "較小"
        info_text = f"""人工標記結果與程式估計值的差異超過容許範圍：

• 人工標記值（含方向）： {measured_displacement:.3f} mm
• 人工標記絕對值： {measured_magnitude:.3f} mm
• 程式估計值： {original_displacement:.3f} mm
• 差異： {direction} {abs(difference_mm):.3f} mm (≈ {difference_px:.2f} 像素)
• 容許差異閾值： {mm_threshold:.3f} mm (≈ {pixel_threshold:.1f} 像素)

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

    def reset_to_first_line_annotation(self):
        """重置回 ROI 圈選階段，清空所有標註記錄"""
        print(f"📝 重置標註狀態：返回 ROI 圈選階段")

        # 重置階段回到 ROI 圈選（讓用戶重新圈選 ROI）
        self.current_phase = "roi_selection"
        self.current_line_index = 0
        self.current_point_in_line = 0
        self.roi_rect = None  # 清除已有的 ROI

        # 清空所有線段標註記錄
        line1_count = len(self.line_annotations[0])
        line2_count = len(self.line_annotations[1])
        self.line_annotations = [[], []]
        self.reference_lines = []
        self.current_line_points = []

        print(f"  - 已清空第一條線段 {line1_count} 次標註")
        print(f"  - 已清空第二條線段 {line2_count} 次標註")
        print(f"  - 重置到 ROI 圈選階段")

        # 清除畫布上的標記
        self.canvas.delete("line_marker")
        self.canvas.delete("existing_line")

        # 重新顯示該群集（根據 current_phase = "roi_selection"，會顯示原始影像供圈選 ROI）
        self.show_current_cluster()

        self.update_status_message()

        print(f"✅ 重置完成，請重新圈選 ROI 區域")

    def save_temporary_state(self) -> str:
        """儲存暫時工作狀態到JSON檔案"""
        from datetime import datetime
        import json
        import os

        # 生成暫存檔案名稱
        csv_path = Path(self.data_manager.csv_path)
        csv_stem = csv_path.stem  # 檔案名稱（不含副檔名）

        # 暫存檔案路徑：與CSV檔案同目錄，格式為 {csv_name}_temp_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{csv_stem}_temp_{timestamp}.json"
        temp_path = csv_path.parent / temp_filename

        # 建立暫存狀態資料結構
        temp_data = {
            "metadata": {
                "csv_file": csv_path.name,
                "csv_path": str(csv_path),
                "video_file": self.jpg_handler.video_name,
                "save_timestamp": datetime.now().isoformat(),
                "format_version": "1.0"
            },
            "progress": {
                "current_cluster_index": self.current_cluster_index,
                "total_clusters": self.data_manager.get_total_clusters(),
                "current_phase": self.current_phase,
                "current_line_index": self.current_line_index,
                "current_point_in_line": self.current_point_in_line
            },
            "settings": {
                "zoom_factor": self.zoom_factor,
                "max_annotations": self.max_annotations
            },
            "current_state": {
                "roi_rect": self.roi_rect,
                "show_reference_lines": self.show_reference_lines,
                "line_annotations": self._serialize_line_annotations(),
                "reference_lines": self._serialize_reference_lines()
            },
            "csv_modifications": self._get_csv_modifications()
        }

        # 寫入JSON檔案
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(temp_data, f, ensure_ascii=False, indent=2)

        print(f"📄 暫存檔案已建立: {temp_path}")
        print(f"   - 進度: {self.current_cluster_index}/{self.data_manager.get_total_clusters()}")
        print(f"   - 當前階段: {self.current_phase}")
        print(f"   - 時間戳: {timestamp}")

        return str(temp_path)

    def _serialize_line_annotations(self) -> list:
        """序列化線段標註資料"""
        serialized = []
        for line_idx, annotations in enumerate(self.line_annotations):
            line_data = []
            for annotation in annotations:
                line_data.append({
                    "timestamp": annotation.timestamp,
                    "start_pixel_coords": annotation.start_pixel_coords,
                    "end_pixel_coords": annotation.end_pixel_coords,
                    "csv_index": annotation.csv_index,
                    "start_roi_coords": annotation.start_roi_coords,
                    "end_roi_coords": annotation.end_roi_coords,
                    "y_component": annotation.y_component,
                    "length": annotation.length
                })
            serialized.append(line_data)
        return serialized

    def _serialize_reference_lines(self) -> list:
        """序列化參考線段資料"""
        serialized = []
        for line in self.reference_lines:
            serialized.append({
                "timestamp": line.timestamp,
                "start_pixel_coords": line.start_pixel_coords,
                "end_pixel_coords": line.end_pixel_coords,
                "csv_index": line.csv_index,
                "start_roi_coords": line.start_roi_coords,
                "end_roi_coords": line.end_roi_coords,
                "y_component": line.y_component,
                "length": line.length
            })
        return serialized

    def _get_csv_modifications(self) -> dict:
        """取得CSV修改記錄"""
        # 記錄已修改的CSV數據（只記錄已完成的群集）
        modifications = {
            "completed_clusters": [],
            "displacement_column": self.data_manager.displacement_column  # displacement column name
        }

        # 記錄每個已完成群集的修改詳情
        for cluster_idx in range(self.current_cluster_index):
            cluster = self.data_manager.get_cluster(cluster_idx)

            # 取得該群集的CSV行範圍
            if hasattr(cluster, 'physical_cluster'):
                physical_cluster = cluster.physical_cluster
                start_row = physical_cluster.pre_zero_index
                end_row = physical_cluster.post_zero_index

                # 記錄修改的行和值
                modified_rows = {}
                for row_idx in range(start_row, end_row + 1):
                    modified_rows[row_idx] = float(self.data_manager.df.iloc[row_idx, self.data_manager.displacement_col_index])

                modifications["completed_clusters"].append({
                    "cluster_index": cluster_idx,
                    "physical_cluster_id": physical_cluster.cluster_id,
                    "csv_row_range": [start_row, end_row],
                    "modified_values": modified_rows
                })

        return modifications

    def load_temporary_state(self, temp_data: dict):
        """載入暫存工作狀態"""
        print(f"📂 載入暫存狀態...")

        try:
            # 恢復進度狀態
            progress = temp_data["progress"]
            self.current_cluster_index = progress["current_cluster_index"]
            self.current_phase = progress["current_phase"]
            self.current_line_index = progress["current_line_index"]
            self.current_point_in_line = progress["current_point_in_line"]

            # 恢復設定
            settings = temp_data["settings"]
            self.zoom_factor = settings.get("zoom_factor", 8)
            self.max_annotations = settings.get("max_annotations", 3)

            # 恢復當前狀態
            current_state = temp_data["current_state"]
            self.roi_rect = current_state.get("roi_rect")
            self.show_reference_lines = current_state.get("show_reference_lines", True)

            # 恢復線段標註
            if current_state.get("line_annotations"):
                self.line_annotations = self._deserialize_line_annotations(current_state["line_annotations"])

            # 恢復參考線段
            if current_state.get("reference_lines"):
                self.reference_lines = self._deserialize_reference_lines(current_state["reference_lines"])

            # 恢復CSV修改
            self._restore_csv_modifications(temp_data["csv_modifications"])

            print(f"   - 進度: 群集 {self.current_cluster_index}/{progress['total_clusters']}")
            print(f"   - 階段: {self.current_phase}")
            print(f"   - 已恢復 {len([anno for line_annos in self.line_annotations for anno in line_annos])} 個線段標註")
            print(f"   - 已恢復 {len(temp_data['csv_modifications']['completed_clusters'])} 個已完成群集的修改")

        except Exception as e:
            print(f"❌ 載入暫存狀態失敗: {e}")
            # 重置為初始狀態
            self.current_cluster_index = 0
            self.current_phase = "roi_selection"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.roi_rect = None
            self.line_annotations = [[], []]
            self.reference_lines = []

    def _deserialize_line_annotations(self, serialized_data: list) -> list:
        """反序列化線段標註資料"""
        line_annotations = []
        for line_data in serialized_data:
            annotations = []
            for annotation_data in line_data:
                annotation = ReferenceLine(
                    timestamp=annotation_data["timestamp"],
                    start_pixel_coords=tuple(annotation_data["start_pixel_coords"]),
                    end_pixel_coords=tuple(annotation_data["end_pixel_coords"]),
                    csv_index=annotation_data["csv_index"],
                    start_roi_coords=tuple(annotation_data["start_roi_coords"]),
                    end_roi_coords=tuple(annotation_data["end_roi_coords"])
                )
                annotations.append(annotation)
            line_annotations.append(annotations)
        return line_annotations

    def _deserialize_reference_lines(self, serialized_data: list) -> list:
        """反序列化參考線段資料"""
        reference_lines = []
        for line_data in serialized_data:
            line = ReferenceLine(
                timestamp=line_data["timestamp"],
                start_pixel_coords=tuple(line_data["start_pixel_coords"]),
                end_pixel_coords=tuple(line_data["end_pixel_coords"]),
                csv_index=line_data["csv_index"],
                start_roi_coords=tuple(line_data["start_roi_coords"]),
                end_roi_coords=tuple(line_data["end_roi_coords"])
            )
            reference_lines.append(line)
        return reference_lines

    def _restore_csv_modifications(self, modifications: dict):
        """恢復CSV修改"""
        completed_clusters = modifications.get("completed_clusters", [])

        for cluster_info in completed_clusters:
            modified_values = cluster_info["modified_values"]
            for row_idx, value in modified_values.items():
                # 恢復CSV中的修改值
                self.data_manager.df.iloc[int(row_idx), self.data_manager.displacement_col_index] = value

        print(f"   - 已恢復 {len(completed_clusters)} 個群集的CSV修改")

    def quit_application(self):
        """退出應用程式"""
        if messagebox.askokcancel("確認退出", "是否要退出校正工具？\n未儲存的更改將丟失。"):
            self.root.quit()

def find_temp_files(csv_path: str) -> list:
    """尋找CSV檔案對應的暫存檔案"""
    import glob
    import json
    from datetime import datetime

    csv_path = Path(csv_path)
    csv_stem = csv_path.stem

    # 搜尋同目錄下的暫存檔案
    temp_pattern = str(csv_path.parent / f"{csv_stem}_temp_*.json")
    temp_files = glob.glob(temp_pattern)

    # 解析並驗證暫存檔案
    valid_temp_files = []
    for temp_file in temp_files:
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)

            # 驗證檔案格式
            if all(key in temp_data for key in ["metadata", "progress", "csv_modifications"]):
                # 解析時間戳
                save_time = datetime.fromisoformat(temp_data["metadata"]["save_timestamp"])
                valid_temp_files.append({
                    "path": temp_file,
                    "data": temp_data,
                    "save_time": save_time,
                    "progress": f"{temp_data['progress']['current_cluster_index']}/{temp_data['progress']['total_clusters']}"
                })
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ 無效的暫存檔案: {temp_file} - {e}")

    # 按時間戳排序（最新的在前）
    valid_temp_files.sort(key=lambda x: x["save_time"], reverse=True)

    return valid_temp_files


def select_temp_file(root: tk.Tk, temp_files: list) -> dict:
    """讓用戶選擇要載入的暫存檔案"""
    if len(temp_files) == 1:
        # 只有一個暫存檔案，直接詢問是否載入
        temp_info = temp_files[0]
        result = messagebox.askyesno(
            "發現暫存檔案",
            f"發現工作暫存檔案：\n\n"
            f"建立時間：{temp_info['save_time'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"進度：{temp_info['progress']} 群集\n"
            f"階段：{temp_info['data']['progress']['current_phase']}\n\n"
            f"是否要載入此暫存狀態？"
        )
        return temp_info if result else None

    # 多個暫存檔案，創建選擇對話框
    dialog = tk.Toplevel(root)
    dialog.title("選擇暫存檔案")
    dialog.geometry("600x400")
    dialog.modal = True
    dialog.grab_set()

    # 置中顯示
    dialog.transient(root)
    x = (dialog.winfo_screenwidth() // 2) - (300)
    y = (dialog.winfo_screenheight() // 2) - (200)
    dialog.geometry(f"600x400+{x}+{y}")

    selected_temp = {"choice": None}

    # 標題
    title_label = ttk.Label(dialog, text="發現多個暫存檔案，請選擇要載入的版本：", font=("Arial", 12, "bold"))
    title_label.pack(pady=10)

    # 列表框架
    list_frame = ttk.Frame(dialog)
    list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # 列表框
    columns = ("時間", "進度", "階段", "檔案")
    tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)

    # 設定欄位標題
    tree.heading("時間", text="建立時間")
    tree.heading("進度", text="進度")
    tree.heading("階段", text="當前階段")
    tree.heading("檔案", text="檔案名稱")

    # 設定欄位寬度
    tree.column("時間", width=150)
    tree.column("進度", width=80)
    tree.column("階段", width=120)
    tree.column("檔案", width=200)

    # 添加資料
    for i, temp_info in enumerate(temp_files):
        tree.insert("", "end", values=(
            temp_info["save_time"].strftime("%Y-%m-%d %H:%M:%S"),
            temp_info["progress"],
            temp_info["data"]["progress"]["current_phase"],
            Path(temp_info["path"]).name
        ), tags=(i,))

    tree.pack(fill=tk.BOTH, expand=True)

    # 按鈕框架
    button_frame = ttk.Frame(dialog)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    def on_load():
        selection = tree.selection()
        if selection:
            item = tree.item(selection[0])
            index = int(tree.item(selection[0], "tags")[0])
            selected_temp["choice"] = temp_files[index]
            dialog.destroy()
        else:
            messagebox.showwarning("請選擇", "請先選擇一個暫存檔案")

    def on_skip():
        selected_temp["choice"] = None
        dialog.destroy()

    ttk.Button(button_frame, text="載入選擇的暫存檔案", command=on_load).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="跳過暫存檔案", command=on_skip).pack(side=tk.LEFT)

    # 等待用戶選擇
    dialog.wait_window()

    return selected_temp["choice"]


def main():
    """主函數 - 選擇檔案並啟動校正工具"""

    # 建立根視窗但隱藏
    root = tk.Tk()
    root.withdraw()
    
    try:
        # 選擇清理後的CSV檔案
        csv_path = filedialog.askopenfilename(
            title="選擇分析結果CSV檔案",
            initialdir="lifts/result",
            filetypes=[("CSV檔案", "*.csv"), ("所有檔案", "*.*")]
        )
        
        if not csv_path:
            return

        # 檢查是否有對應的暫存檔案
        temp_data = None
        temp_files = find_temp_files(csv_path)
        if temp_files:
            temp_data = select_temp_file(root, temp_files)

        # 從CSV檔名推導影片檔名（僅用於查找JPG目錄）
        csv_filename = Path(csv_path).name
        # 支援帶前綴或不帶前綴的CSV檔案
        if csv_filename.startswith('c'):
            video_filename = csv_filename[1:]  # 移除 'c' 前綴
            video_filename = video_filename.replace('.csv', '.mp4')
        elif csv_filename.startswith('mc'):
            video_filename = csv_filename[2:]  # 移除 'mc' 前綴
            video_filename = video_filename.replace('.csv', '.mp4')
        else:
            # 不帶前綴的CSV檔案，直接使用檔名
            video_filename = csv_filename.replace('.csv', '.mp4')
        
        print(f"準備處理:")
        print(f"CSV檔案: {csv_path}")
        print(f"預期JPG目錄: lifts/exported_frames/{Path(video_filename).stem}/")
        
        # 初始化數據管理器
        data_manager = DataManager(csv_path, video_filename)
        
        if data_manager.get_total_clusters() == 0:
            messagebox.showinfo("完成", "此檔案沒有需要校正的位移群集")
            return
        
        print(f"發現 {data_manager.get_total_clusters()} 個需要校正的位移群集")
        
        # 初始化JPG處理器
        jpg_handler = JPGHandler(video_filename)
        
        # 啟動校正界面
        app = CorrectionApp(
            root,
            data_manager,
            jpg_handler,
        )

        # 如果有暫存資料，載入狀態
        if temp_data:
            app.load_temporary_state(temp_data["data"])
            print(f"✅ 已載入暫存狀態：進度 {temp_data['progress']}")

        app.start_correction()
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("錯誤", f"初始化失敗: {str(e)}")
    finally:
        root.destroy()

if __name__ == '__main__':
    main()
