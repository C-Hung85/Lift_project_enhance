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
        
        # 識別所有需要校正的群集
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

    def get_frame_at_index(self, frame_number: int) -> Optional[np.ndarray]:
        """獲取指定幀號的影片幀"""
        print(f"影片幀提取: 幀號={frame_number}")
        
        if frame_number >= self.total_frames:
            print(f"錯誤: 幀號 {frame_number} 超出範圍 (總幀數: {self.total_frames})")
            return None
        
        success = False
        max_attempts = 3
        for attempt in range(max_attempts):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(f"嘗試 {attempt+1}: 設置幀位置 目標={frame_number}, 實際={int(actual_pos)}")
            
            if abs(actual_pos - frame_number) < 1:
                ret, frame = self.cap.read()
                if ret:
                    print(f"成功讀取幀 {frame_number} (嘗試 {attempt+1})")
                    success = True
                    break
            
            print(f"重置 VideoCapture (嘗試 {attempt+1})")
            self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not success:
            print("使用全新 VideoCapture 重試...")
            temp_cap = cv2.VideoCapture(self.video_path)
            temp_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = temp_cap.read()
            temp_cap.release()
            if not ret:
                print(f"錯誤: 無法讀取幀 {frame_number}")
                return None
        
        if self.rotation_angle != 0:
            frame = rotate_frame(frame, self.rotation_angle)
        
        print(f"成功提取幀 {frame_number}, 尺寸: {frame.shape}")
        return frame

    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """獲取指定時戳的影片幀 (舊版，可能有偏差)"""
        frame_number = int(timestamp * self.fps)
        print(f"影片幀提取 (估算): 時戳={timestamp:.3f}s → 幀號={frame_number}")
        return self.get_frame_at_index(frame_number)
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

class CorrectionApp:
    """半自動校正GUI應用程式"""
    
    def __init__(self, root: tk.Tk, data_manager: DataManager, video_handler: VideoHandler):
        self.root = root
        self.data_manager = data_manager
        self.video_handler = video_handler
        
        # 校正狀態
        self.current_cluster_index = 0
        self.current_phase = "roi_selection"  # roi_selection, line_marking_1, line_marking_2
        self.current_line_index = 0  # 0: 第一條線段, 1: 第二條線段
        self.current_point_in_line = 0  # 0: 線段起點, 1: 線段終點
        self.reference_lines = []  # 儲存當前群集的參考線段
        self.current_line_points = []  # 儲存當前正在標記的線段點 [(x1,y1), (x2,y2)]
        self.roi_rect = None  # (x, y, width, height)
        self.zoom_factor = 4
        
        # GUI 組件
        self.setup_ui()
        
        # 鍵盤綁定
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
    def setup_ui(self):
        """設置使用者界面"""
        self.root.deiconify()
        self.root.title("半自動位移校正工具")
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
        
        self.help_label = ttk.Label(status_frame, text="快捷鍵: [N]ext [B]ack [S]ave [Q]uit", font=("Arial", 9))
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
        
        # 檢查是否有前零點
        has_pre_zero = getattr(cluster, 'has_pre_zero', True)
        
        # 決定要顯示的時戳
        if self.current_phase in ["roi_selection", "line_marking_1"]:
            if has_pre_zero:
                timestamp = cluster.timestamps[0]  # 前零點
                description = "群集前零點 (第一條線段)"
            else:
                timestamp = cluster.timestamps[0]  # 群集開始點（第一行就有位移）
                description = "群集開始點 (檔案開頭)"
        else:  # line_marking_2
            timestamp = cluster.timestamps[-1]  # 群集結束點
            description = "群集結束點 (第二條線段)"
            
        # 添加調試信息
        print(f"\n=== 時戳調試信息 ===")
        print(f"當前階段: {self.current_phase}")
        print(f"群集索引: {cluster.start_index} 到 {cluster.end_index}")
        print(f"有前零點: {has_pre_zero}")
        print(f"時戳數組: {cluster.timestamps}")
        print(f"選中時戳: {timestamp:.6f}s (索引: {'0' if self.current_phase in ['roi_selection', 'line_marking_1'] else '-1'})")
        print(f"時戳差異: {cluster.timestamps[-1] - cluster.timestamps[0]:.6f}s")
        if len(cluster.original_values) > 0:
            print(f"原始位移值: {cluster.original_values}")
            print(f"位移總和: {sum(abs(v) for v in cluster.original_values):.3f}")
        print("===================")
        
        # 更新資訊
        total_clusters = self.data_manager.get_total_clusters()
        cluster_info = f"檔案: {self.video_handler.video_name} | "
        cluster_info += f"群集: {self.current_cluster_index + 1}/{total_clusters} | "
        cluster_info += f"時戳: {timestamp:.3f}s | "
        cluster_info += f"標記點: {description}"
        
        self.info_label.config(text=cluster_info)
        
        # 顯示影片幀
        frame = self.video_handler.get_frame_at_timestamp(timestamp)
        if frame is None:
            messagebox.showerror("錯誤", f"無法獲取時戳 {timestamp:.3f}s 的影片幀")
            return
        
        self.show_frame(frame)
        
        # 更新狀態
        if self.current_phase == "roi_selection":
            if not has_pre_zero:
                self.status_label.config(text="⚠️ 故障檢測: 檔案開頭即有位移，請檢視畫面後按 [N] 選擇處理方式")
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
        
        # 放大4倍
        enlarged_roi = cv2.resize(roi_frame, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        # 只顯示放大的ROI，不要更新 original_frame
        self.display_frame_only(enlarged_roi)
    
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
            # 第一個點 - 清除之前的標記並開始新線段
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
            
            # 繪製連接線
            start_canvas_coords = self.pixel_to_canvas_coords(self.current_line_points[0])
            end_canvas_coords = self.pixel_to_canvas_coords(self.current_line_points[1])
            
            if start_canvas_coords and end_canvas_coords:
                self.canvas.create_line(
                    start_canvas_coords[0], start_canvas_coords[1],
                    end_canvas_coords[0], end_canvas_coords[1],
                    fill="lime", width=3, tags="line_marker"
                )
            
            # 儲存完整的線段
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
            
            # 儲存或替換線段
            if self.current_line_index < len(self.reference_lines):
                self.reference_lines[self.current_line_index] = line
            else:
                self.reference_lines.append(line)
            
            # 重置線段標記狀態
            self.current_point_in_line = 0
            self.current_line_points = []
            
            self.update_status_message()
    
    def update_status_message(self):
        """更新狀態提示訊息"""
        if self.current_phase == "roi_selection":
            self.status_label.config(text="階段1: 請拖拽選擇 ROI 區域，完成後按 [N] 確認")
        elif self.current_phase == "line_marking_1":
            if self.current_point_in_line == 0:
                self.status_label.config(text="階段2a: 請點擊第一條參考線段的起點")
            else:
                self.status_label.config(text="階段2b: 請點擊第一條參考線段的終點，完成後按 [N] 確認")
        elif self.current_phase == "line_marking_2":
            if self.current_point_in_line == 0:
                self.status_label.config(text="階段3a: 請在第二張畫面的相同結構上標記第二條線段的起點")
            else:
                self.status_label.config(text="階段3b: 請點擊第二條線段的終點，完成後按 [N] 確認並計算位移")
            
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
            if self.current_point_in_line != 0 or len(self.reference_lines) == 0:
                messagebox.showwarning("警告", "請先完成第一條線段的標記")
                return
            
            cluster = self.data_manager.get_cluster(self.current_cluster_index)
            has_pre_zero = getattr(cluster, 'has_pre_zero', True)
            
            if has_pre_zero:
                # 有前零點，移動到第二條線段（群集結束點）
                self.current_phase = "line_marking_2"
                self.current_line_index = 1
                self.current_point_in_line = 0
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
            if self.current_point_in_line != 0 or len(self.reference_lines) < 2:
                messagebox.showwarning("警告", "請先完成第二條線段的標記")
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
            if len(self.reference_lines) > 1:
                self.reference_lines.pop()  # 移除第二條線段
            self.show_current_cluster()
            self.enter_precision_marking_mode()
            self.update_status_message()
        elif self.current_phase == "line_marking_1":
            # 從第一條線段回到ROI選擇
            self.current_phase = "roi_selection"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.reference_lines = []
            self.roi_rect = None
            self.show_current_cluster()
        elif self.current_cluster_index > 0:
            # 回到上一個群集
            self.current_cluster_index -= 1
            self.current_phase = "roi_selection"
            self.current_line_index = 0
            self.current_point_in_line = 0
            self.reference_lines = []
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
        
        # 顯示線段詳細資訊
        print(f"\n=== 線段校正計算 ===")
        print(f"第一條線段:")
        print(f"  起點: {line1.start_pixel_coords}")
        print(f"  終點: {line1.end_pixel_coords}")
        print(f"  Y分量: {line1.y_component:.1f} 像素")
        print(f"第二條線段:")
        print(f"  起點: {line2.start_pixel_coords}")
        print(f"  終點: {line2.end_pixel_coords}")
        print(f"  Y分量: {line2.y_component:.1f} 像素")
        print(f"差異計算:")
        print(f"  Y分量差異: {line2.y_component:.1f} - {line1.y_component:.1f} = {line2.y_component - line1.y_component:.1f} 像素")
        print(f"  比例尺: {self.data_manager.scale_factor} 像素/10mm")
        print(f"  計算位移: ({line2.y_component - line1.y_component:.1f} × 10) / {self.data_manager.scale_factor} = {measured_displacement:.3f} mm")
        print("=====================")
        
        # 應用校正
        cluster = self.data_manager.get_cluster(self.current_cluster_index)
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
    
    def quit_application(self):
        """退出應用程式"""
        if messagebox.askokcancel("確認退出", "是否要退出校正工具？\n未儲存的更改將丟失。"):
            self.root.quit()

def main():
    """主函數 - 選擇檔案並啟動校正工具"""
    
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
        app = CorrectionApp(root, data_manager, video_handler)
        app.start_correction()
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("錯誤", f"初始化失敗: {str(e)}")
    finally:
        root.destroy()

if __name__ == '__main__':
    main()
