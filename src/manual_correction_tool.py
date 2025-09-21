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
    original_values: List[float]  # 原始位移值 [start, ..., end]
    csv_indices: List[int]        # CSV中的行號 [pre_zero, start, ..., end]

@dataclass
class ReferencePoint:
    """參考點數據結構"""
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
        
        if self.scale_factor is None:
            raise ValueError(f"找不到影片 {video_name} 的比例尺配置")
        
        # 識別所有需要校正的群集
        self.clusters = self._identify_clusters()
        
    def _identify_clusters(self) -> List[CorrectionCluster]:
        """識別所有需要校正的非零值群集"""
        clusters = []
        displacement_col = self.df.columns[1]  # 第二欄是位移數據
        
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
                
                # 建立時戳列表
                if has_pre_zero:
                    timestamps = [
                        self.df.iloc[pre_zero_idx]['second'],  # 前零點時戳
                        *[self.df.iloc[j]['second'] for j in range(start_idx, end_idx + 1)]  # 群集時戳
                    ]
                    csv_indices = list(range(pre_zero_idx, end_idx + 1))
                else:
                    # 沒有前零點，直接使用群集範圍
                    timestamps = [self.df.iloc[j]['second'] for j in range(start_idx, end_idx + 1)]
                    csv_indices = list(range(start_idx, end_idx + 1))
                
                # 建立群集
                cluster = CorrectionCluster(
                    start_index=start_idx,
                    end_index=end_idx,
                    pre_zero_index=pre_zero_idx,
                    timestamps=timestamps,
                    original_values=[
                        self.df.iloc[j][displacement_col] for j in range(start_idx, end_idx + 1)
                    ],
                    csv_indices=csv_indices
                )
                
                # 為特殊情況添加標記
                cluster.has_pre_zero = has_pre_zero
                
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
    
    def calculate_displacement(self, point1: ReferencePoint, point2: ReferencePoint) -> float:
        """
        計算兩個參考點之間的實際位移 (mm)
        
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
        
        # 計算最小位移閾值 (基於比例尺的50%)
        min_displacement_threshold = (10.0 / self.scale_factor) * 0.5  # 0.5像素對應的mm
        
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
    
    def get_frame_at_timestamp(self, timestamp: float) -> Optional[np.ndarray]:
        """
        獲取指定時戳的影片幀
        
        Args:
            timestamp: 時戳 (秒)
            
        Returns:
            影片幀，如果失敗則返回 None
        """
        frame_number = int(timestamp * self.fps)
        
        if frame_number >= self.total_frames:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        # 應用旋轉校正
        if self.rotation_angle != 0:
            frame = rotate_frame(frame, self.rotation_angle)
        
        return frame
    
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
        self.current_phase = "roi_selection"  # roi_selection, precision_marking
        self.current_point_index = 0  # 0: 第一個點, 1: 第二個點
        self.reference_points = []  # 儲存當前群集的參考點
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
        if self.current_point_index == 0:
            self.current_phase = "roi_selection"
            self.reference_points = []
            self.roi_rect = None
        else:
            self.current_phase = "roi_selection"
        
        # 檢查是否有前零點
        has_pre_zero = getattr(cluster, 'has_pre_zero', True)
        
        # 決定要顯示的時戳
        if self.current_point_index == 0:
            if has_pre_zero:
                timestamp = cluster.timestamps[0]  # 前零點
                description = "群集前零點"
            else:
                timestamp = cluster.timestamps[0]  # 群集開始點（第一行就有位移）
                description = "群集開始點 (檔案開頭)"
        else:
            timestamp = cluster.timestamps[-1]  # 群集結束點
            description = "群集結束點"
        
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
            if not has_pre_zero and self.current_point_index == 0:
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
        
    def on_canvas_click(self, event):
        """滑鼠點擊事件"""
        if self.current_phase == "roi_selection":
            self.drag_start = (event.x, event.y)
            
        elif self.current_phase == "precision_marking":
            # 精細標記模式
            self.place_reference_point(event.x, event.y)
    
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
            
            # 進入精細標記模式
            self.enter_precision_marking_mode()
            
            self.drag_start = None
    
    def enter_precision_marking_mode(self):
        """進入精細標記模式"""
        self.current_phase = "precision_marking"
        
        # 提取ROI並放大
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        roi_frame = self.original_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # 放大4倍
        enlarged_roi = cv2.resize(roi_frame, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        self.show_frame(enlarged_roi)
        
        # 更新狀態
        self.status_label.config(text="階段2: 請點擊標記參考點位置，完成後按 [N] 確認")
    
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
        
        if self.current_point_index == 0:
            timestamp = cluster.timestamps[0]
            csv_index = cluster.csv_indices[0]
        else:
            timestamp = cluster.timestamps[-1]
            csv_index = cluster.csv_indices[-1]
        
        reference_point = ReferencePoint(
            timestamp=timestamp,
            pixel_coords=(original_x, original_y),
            csv_index=csv_index,
            roi_coords=(roi_local_x, roi_local_y)
        )
        
        # 更新或添加參考點
        if len(self.reference_points) <= self.current_point_index:
            self.reference_points.append(reference_point)
        else:
            self.reference_points[self.current_point_index] = reference_point
        
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
        if self.current_phase == "precision_marking":
            if len(self.reference_points) <= self.current_point_index:
                messagebox.showwarning("警告", "請先標記參考點")
                return
            
            cluster = self.data_manager.get_cluster(self.current_cluster_index)
            has_pre_zero = getattr(cluster, 'has_pre_zero', True)
            
            if self.current_point_index == 0:
                if has_pre_zero:
                    # 有前零點，移動到第二個點（群集結束點）
                    self.current_point_index = 1
                    self.show_current_cluster()
                else:
                    # 沒有前零點，可能是故障，提供清零選項
                    self.handle_first_line_displacement()
                    return
            else:
                # 兩個點都已標記，計算並應用校正
                self.apply_cluster_correction()
                
                # 移動到下一個群集
                self.move_to_next_cluster()
    
    def previous_step(self):
        """返回上一步"""
        if self.current_point_index > 0:
            self.current_point_index -= 1
            self.show_current_cluster()
        elif self.current_cluster_index > 0:
            self.current_cluster_index -= 1
            self.current_point_index = 1
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
        
        self.current_point_index = 0
        self.show_current_cluster()

    def apply_cluster_correction(self):
        """應用當前群集的校正"""
        if len(self.reference_points) < 2:
            messagebox.showerror("錯誤", "需要兩個參考點才能計算位移")
            return
        
        # 計算實際位移
        point1 = self.reference_points[0]  # 前零點
        point2 = self.reference_points[1]  # 結束點
        
        measured_displacement = self.data_manager.calculate_displacement(point1, point2)
        
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
