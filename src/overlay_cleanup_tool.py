#!/usr/bin/env python3
"""
疑似雜訊疊圖驗證工具

用於協助檢視人工校正（mc*.csv）後仍殘留的小幅運動群集，
透過兩階段疊圖法確認群集是否為純雜訊，並可直接將其清零。
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np

# 將 src 目錄加入路徑以復用人工校正工具的模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from manual_correction_tool import (  # type: ignore  # pylint: disable=wrong-import-position
    DataManager,
    JPGHandler,
)


@dataclass
class SuspiciousCluster:
    """疑似雜訊群集描述"""

    cluster_index: int                     # 在 DataManager.clusters 中的索引
    non_zero_count: int                    # 非零點數
    span_rows: int                         # 群集覆蓋的列數
    max_abs_value: float                   # 最大位移絕對值
    total_abs_value: float                 # 位移絕對值總和


def derive_video_filename(csv_path: Path) -> str:
    """
    由 mc*.csv 推導影片檔名（維持與人工校正工具相同的命名規則）
    """
    name = csv_path.name
    if name.startswith("mc"):
        base_name = name[2:]
    else:
        base_name = name
    return Path(base_name).with_suffix(".mp4").name


def find_suspicious_clusters(data_manager: DataManager) -> List[SuspiciousCluster]:
    """
    掃描資料尋找疑似雜訊群集

    規則：
        1. 群集非零點數 ≤ 3
        2. 若群集橫跨 4 個（含）以上資料列則排除
        3. 若任一點的位移 < 0.1mm 或群集總位移 < 0.2mm 則視為疑似
    """
    suspicious: List[SuspiciousCluster] = []

    for idx, cluster in enumerate(data_manager.clusters):
        physical = getattr(cluster, "physical_cluster", None)
        if physical is None:
            continue

        # 取得非零位移值
        values: List[float] = []
        for row_index in range(cluster.start_index, cluster.end_index + 1):
            value = data_manager.df.iloc[row_index, data_manager.displacement_col_index]
            if value != 0:
                values.append(float(value))

        if not values:
            continue  # 已被清零

        non_zero_count = len(values)
        if non_zero_count > 3:
            continue

        span_rows = cluster.end_index - cluster.start_index + 1
        if span_rows >= 4:
            continue

        abs_values = [abs(v) for v in values]
        max_abs_value = max(abs_values)
        total_abs_value = sum(abs_values)

        one_pixel_mm = one_pixel_in_mm(data_manager.scale_factor)

        if (
            max_abs_value < 0.1
            or total_abs_value <= one_pixel_mm
            or any(v < 0.1 for v in abs_values)
        ):
            suspicious.append(
                SuspiciousCluster(
                    cluster_index=idx,
                    non_zero_count=non_zero_count,
                    span_rows=span_rows,
                    max_abs_value=max_abs_value,
                    total_abs_value=total_abs_value,
                )
            )

    return suspicious


def translate_image(image: np.ndarray, shift: Tuple[float, float]) -> np.ndarray:
    """
    將影像平移指定的 (dx, dy)（支援亞像素位移）
    """
    dx, dy = shift
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return shifted


def one_pixel_in_mm(scale_factor: float) -> float:
    """計算相當於 1 像素的實際距離 (mm)"""
    if scale_factor == 0:
        return 0.0
    return 10.0 / scale_factor


def compute_contrast_score(
    base: np.ndarray, overlay: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """
    計算疊圖對比指標（採用歸一化互相關）
    返回值介於 [-1, 1]，越接近 1 代表混合後對齊程度越高。
    """
    if base.size == 0 or overlay.size == 0:
        return 0.0

    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if mask is not None:
        if mask.shape != base_gray.shape:
            raise ValueError("遮罩尺寸須與影像相同")
        mask = mask.astype(bool)
        base_values = base_gray[mask]
        overlay_values = overlay_gray[mask]
    else:
        base_values = base_gray.reshape(-1)
        overlay_values = overlay_gray.reshape(-1)

    if base_values.size == 0 or overlay_values.size == 0:
        return 0.0

    base_mean = base_values.mean()
    overlay_mean = overlay_values.mean()

    base_centered = base_values - base_mean
    overlay_centered = overlay_values - overlay_mean

    numerator = float(np.sum(base_centered * overlay_centered))
    denominator = float(
        np.sqrt(np.sum(base_centered ** 2) * np.sum(overlay_centered ** 2))
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


class OverlayCleanupApp:
    """疊圖檢視與雜訊清理 GUI"""

    def __init__(
        self,
        root: tk.Tk,
        data_manager: DataManager,
        jpg_handler: JPGHandler,
        suspicious_clusters: List[SuspiciousCluster],
    ):
        self.root = root
        self.data_manager = data_manager
        self.jpg_handler = jpg_handler
        self.suspicious_clusters = suspicious_clusters

        # 介面狀態
        self.current_index = 0
        self.phase = "roi_selection"  # roi_selection, global_alignment, split_line, split_alignment, decision

        # 圖像資料
        self.pre_frame: Optional[np.ndarray] = None
        self.post_frame: Optional[np.ndarray] = None
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None
        self.roi_pre: Optional[np.ndarray] = None
        self.roi_post: Optional[np.ndarray] = None
        self.current_overlay: Optional[np.ndarray] = None

        # 平移參數
        self.global_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.right_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.split_line_points_roi: List[Tuple[float, float]] = []
        self.split_line_preview_roi: Optional[Tuple[float, float]] = None
        self.split_mask: Optional[np.ndarray] = None

        # 對比追蹤
        self.global_contrast_best: Optional[float] = None
        self.global_contrast_current: Optional[float] = None
        self.split_contrast_best_left: Optional[float] = None
        self.split_contrast_best_right: Optional[float] = None
        self.split_contrast_current_left: Optional[float] = None
        self.split_contrast_current_right: Optional[float] = None
        self.latest_measured_mm: Optional[float] = None
        self.latest_csv_mm: Optional[float] = None

        # Tk 組件
        self.setup_ui()
        self.bind_events()

        # 用於畫布可視化
        self.canvas_image = None
        self.canvas_bounds = (0, 0, 0, 0)
        self.canvas_rect_id: Optional[int] = None
        self.display_scale = 1.0
        self.drag_start: Optional[Tuple[int, int]] = None

        self.load_next_cluster()

    # ------------------------------------------------------------------ #
    # UI 建置
    # ------------------------------------------------------------------ #
    def setup_ui(self):
        self.root.deiconify()
        self.root.title("疊圖清理工具 - 載入中...")
        self.root.geometry("1280x900")

        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=6, pady=4)

        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.info_label.pack(side=tk.LEFT)

        self.status_label = ttk.Label(info_frame, text="", font=("Arial", 10))
        self.status_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, bg="black", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        help_frame = ttk.Frame(self.root)
        help_frame.pack(fill=tk.X, padx=6, pady=4)

        help_text = (
            "操作提示：ROI完成後按 Enter 進入疊圖；"
            "平移鍵 q/w/e/r=±10px, a/s/d/f=±1px, z/x/c/v=±0.5px；"
            "L 開啟切割線；Enter 完成；M 標記雜訊；U 更新疊圖位移；K 保留；B 返回ROI。"
        )
        ttk.Label(help_frame, text=help_text, font=("Arial", 9)).pack(side=tk.LEFT)

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.root.bind("<Key>", self.on_key_press)
        self.root.focus_set()

    # ------------------------------------------------------------------ #
    # 事件處理
    # ------------------------------------------------------------------ #
    def on_canvas_click(self, event):
        if self.phase == "roi_selection":
            self.drag_start = (event.x, event.y)
        elif self.phase == "split_line":
            self.handle_split_line_click(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.phase == "roi_selection" and self.drag_start:
            if self.canvas_rect_id:
                self.canvas.delete(self.canvas_rect_id)

            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y

            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])

            self.canvas_rect_id = self.canvas.create_rectangle(
                left, top, right, bottom, outline="red", width=2, dash=(6, 6)
            )
        elif self.phase == "split_line" and self.split_line_points_roi:
            roi_point = self.canvas_to_roi_coordinates(event.x, event.y)
            if roi_point is not None:
                self.split_line_preview_roi = roi_point
                self.draw_split_line()

    def on_canvas_release(self, event):
        if self.phase == "roi_selection" and self.drag_start:
            x1, y1 = self.drag_start
            x2, y2 = event.x, event.y
            self.drag_start = None

            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])

            self.set_roi_from_canvas(left, top, right, bottom)

    def on_key_press(self, event):
        key = event.keysym.lower()

        if key == "return":
            self.handle_enter()
            return
        if key == "b":
            self.reset_to_roi_selection()
            return
        if key == "l":
            self.enter_split_line_mode()
            return
        if key == "escape":
            self.cancel_split_line()
            return
        if key == "m":
            self.mark_current_cluster_as_noise()
            return
        if key == "u":
            self.apply_overlay_measurement()
            return
        if key == "k":
            self.keep_current_cluster()
            return

        if self.phase in {"global_alignment", "split_alignment"}:
            self.handle_translation_key(key)

    # ------------------------------------------------------------------ #
    # 階段切換
    # ------------------------------------------------------------------ #
    def handle_enter(self):
        if self.phase == "roi_selection":
            if not self.roi_rect:
                messagebox.showwarning("提醒", "請先使用滑鼠選擇 ROI。")
                return
            self.enter_global_alignment()
        elif self.phase == "global_alignment":
            self.enter_decision_phase()
        elif self.phase == "split_alignment":
            self.enter_decision_phase()
        elif self.phase == "decision":
            self.keep_current_cluster()

    def enter_global_alignment(self):
        if not self.prepare_roi():
            return
        self.phase = "global_alignment"
        self.update_overlay_display()
        self.update_status()

    def enter_split_line_mode(self):
        if self.phase != "global_alignment":
            return
        self.phase = "split_line"
        self.split_line_points_roi = []
        self.split_line_preview_roi = None
        self.split_mask = None
        self.canvas.delete("split_line")
        self.update_status("請在畫面上點擊兩點建立切割線。")

    def cancel_split_line(self):
        if self.phase in {"split_line", "split_alignment"}:
            self.phase = "global_alignment"
            self.split_line_points_roi = []
            self.split_line_preview_roi = None
            self.split_mask = None
            self.right_shift = np.array([0.0, 0.0], dtype=np.float32)
            self.canvas.delete("split_line")
            self.update_overlay_display()
            self.update_status()

    def enter_split_alignment(self):
        if len(self.split_line_points_roi) != 2:
            return
        if not self.prepare_split_mask():
            return
        self.phase = "split_alignment"
        self.right_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.split_line_preview_roi = None
        self.split_contrast_best_left = None
        self.split_contrast_best_right = None
        self.update_overlay_display()
        self.update_status()

    def enter_decision_phase(self):
        self.phase = "decision"
        self.latest_measured_mm = self.calculate_measured_displacement_mm()
        self.latest_csv_mm = self.calculate_current_csv_displacement()
        self.update_status()

    def reset_to_roi_selection(self):
        self.phase = "roi_selection"
        self.roi_rect = None
        self.roi_pre = None
        self.roi_post = None
        self.global_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.right_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.split_line_points_roi = []
        self.split_line_preview_roi = None
        self.split_mask = None
        self.global_contrast_best = None
        self.global_contrast_current = None
        self.split_contrast_best_left = None
        self.split_contrast_best_right = None
        self.split_contrast_current_left = None
        self.split_contrast_current_right = None
        self.canvas.delete("split_line")
        self.canvas_rect_id = None
        self.current_overlay = None
        self.latest_measured_mm = None
        self.latest_csv_mm = None
        self.update_canvas_with_frame(self.pre_frame)
        self.update_status("請使用滑鼠框選 ROI，完成後按 Enter。")

    # ------------------------------------------------------------------ #
    # 叢集管理
    # ------------------------------------------------------------------ #
    def load_next_cluster(self):
        if self.current_index >= len(self.suspicious_clusters):
            messagebox.showinfo("完成", "沒有更多疑似雜訊群集。")
            self.save_results_and_exit()
            return

        target = self.suspicious_clusters[self.current_index]
        cluster = self.data_manager.get_cluster(target.cluster_index)
        physical = getattr(cluster, "physical_cluster", None)
        if physical is None:
            self.current_index += 1
            self.load_next_cluster()
            return

        # 載入對應的前後 JPG
        pre_jpg = physical.pre_zero_jpg
        post_jpg = physical.post_zero_jpg
        self.pre_frame = self.jpg_handler.load_jpg_frame(pre_jpg)
        self.post_frame = self.jpg_handler.load_jpg_frame(post_jpg)

        if self.pre_frame is None or self.post_frame is None:
            messagebox.showerror(
                "錯誤", f"無法載入群集 {physical.cluster_id} 的前後 JPG。\n請確認檔案存在。"
            )
            self.current_index += 1
            self.load_next_cluster()
            return

        self.root.title(
            f"疊圖清理工具 - 群集 {self.current_index + 1}/{len(self.suspicious_clusters)} "
            f"(ID: {physical.cluster_id})"
        )

        pixel_threshold_mm = one_pixel_in_mm(self.data_manager.scale_factor)
        info_text = (
            f"群集 ID {physical.cluster_id} | 非零點數 {target.non_zero_count} | "
            f"行數 {target.span_rows} | max {target.max_abs_value:.3f}mm | "
            f"sum {target.total_abs_value:.3f}mm | 1px≈{pixel_threshold_mm:.3f}mm"
        )
        self.info_label.config(text=info_text)

        self.reset_to_roi_selection()

    def advance_to_next_cluster(self):
        self.current_index += 1
        self.load_next_cluster()

    # ------------------------------------------------------------------ #
    # ROI 與畫布關聯
    # ------------------------------------------------------------------ #
    def set_roi_from_canvas(self, left, top, right, bottom):
        if self.pre_frame is None:
            return

        canvas_x, canvas_y, canvas_w, canvas_h = self.canvas_bounds
        if canvas_w == 0 or canvas_h == 0:
            return

        left = max(canvas_x, left)
        right = min(canvas_x + canvas_w, right)
        top = max(canvas_y, top)
        bottom = min(canvas_y + canvas_h, bottom)

        if right - left < 40 or bottom - top < 40:
            messagebox.showwarning("提醒", "ROI 面積過小，請重新選取。")
            return

        scale = self.display_scale
        roi_x = int((left - canvas_x) / scale)
        roi_y = int((top - canvas_y) / scale)
        roi_w = int((right - left) / scale)
        roi_h = int((bottom - top) / scale)
        self.roi_rect = (roi_x, roi_y, roi_w, roi_h)
        self.update_status("ROI 選取完成，可按 Enter 進入疊圖。")

    def prepare_roi(self) -> bool:
        if self.roi_rect is None or self.pre_frame is None or self.post_frame is None:
            return False

        x, y, w, h = self.roi_rect
        self.roi_pre = self.pre_frame[y : y + h, x : x + w]
        self.roi_post = self.post_frame[y : y + h, x : x + w]
        self.global_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.right_shift = np.array([0.0, 0.0], dtype=np.float32)
        self.global_contrast_best = None
        self.global_contrast_current = None
        self.split_contrast_best_left = None
        self.split_contrast_best_right = None
        self.split_contrast_current_left = None
        self.split_contrast_current_right = None
        return True

    def prepare_split_mask(self) -> bool:
        if len(self.split_line_points_roi) != 2 or self.roi_pre is None:
            return False
        (x1, y1), (x2, y2) = self.split_line_points_roi
        height, width = self.roi_pre.shape[:2]
        yy, xx = np.indices((height, width))
        line_dx = x2 - x1
        line_dy = y2 - y1
        cross = (xx - x1) * line_dy - (yy - y1) * line_dx
        self.split_mask = cross <= 0
        return True

    # ------------------------------------------------------------------ #
    # 畫布與疊圖顯示
    # ------------------------------------------------------------------ #
    def update_canvas_with_frame(self, frame: Optional[np.ndarray]):
        if frame is None:
            self.canvas.delete("all")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.canvas.update_idletasks()
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        h, w = frame_rgb.shape[:2]
        # 允許放大最多 4 倍，且長寬不超過 1440 像素
        max_scale_limit = min(4.0, 1440.0 / max(w, 1), 1440.0 / max(h, 1))
        scale = min(canvas_width / max(w, 1), canvas_height / max(h, 1), max_scale_limit)
        scale = max(scale, 1e-6)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame_rgb, (new_w, new_h))

        from PIL import Image, ImageTk

        image = Image.fromarray(resized)
        self.canvas_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        offset_x = (canvas_width - new_w) // 2
        offset_y = (canvas_height - new_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.canvas_image)
        self.canvas_bounds = (offset_x, offset_y, new_w, new_h)
        self.display_scale = scale

    def update_overlay_display(self):
        if self.roi_pre is None or self.roi_post is None:
            return

        shifted_global = translate_image(self.roi_post, tuple(self.global_shift))
        if self.phase == "split_alignment" and self.split_mask is not None:
            shifted_right = translate_image(
                self.roi_post, tuple(self.global_shift + self.right_shift)
            )
            combined = shifted_global.copy()
            combined[~self.split_mask] = shifted_right[~self.split_mask]

            self.split_contrast_current_left = compute_contrast_score(
                self.roi_pre, combined, mask=self.split_mask
            )
            self.split_contrast_current_right = compute_contrast_score(
                self.roi_pre, combined, mask=~self.split_mask
            )

            if (
                self.split_contrast_best_left is None
                or self.split_contrast_current_left > self.split_contrast_best_left
            ):
                self.split_contrast_best_left = self.split_contrast_current_left
            if (
                self.split_contrast_best_right is None
                or self.split_contrast_current_right > self.split_contrast_best_right
            ):
                self.split_contrast_best_right = self.split_contrast_current_right

            overlay = cv2.addWeighted(self.roi_pre, 0.5, combined, 0.5, 0)
        else:
            overlay = cv2.addWeighted(self.roi_pre, 0.5, shifted_global, 0.5, 0)

            self.global_contrast_current = compute_contrast_score(
                self.roi_pre, shifted_global
            )
            if (
                self.global_contrast_best is None
                or self.global_contrast_current > self.global_contrast_best
            ):
                self.global_contrast_best = self.global_contrast_current

        self.current_overlay = overlay
        self.update_canvas_with_frame(overlay)
        self.draw_split_line()
        self.update_status()

    def draw_split_line(self):
        self.canvas.delete("split_line")
        if not self.split_line_points_roi:
            return

        canvas_points: List[Tuple[int, int]] = []
        for roi_point in self.split_line_points_roi:
            converted = self.roi_to_canvas_coordinates(*roi_point)
            if converted is not None:
                canvas_points.append(converted)

        preview_canvas = None
        if self.split_line_preview_roi is not None:
            preview_canvas = self.roi_to_canvas_coordinates(*self.split_line_preview_roi)

        if not canvas_points:
            return

        for x, y in canvas_points:
            self.canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill="cyan",
                outline="white",
                width=2,
                tags="split_line",
            )

        if len(canvas_points) >= 2:
            (x1, y1), (x2, y2) = canvas_points[:2]
            self.canvas.create_line(
                x1, y1, x2, y2, fill="yellow", width=2, dash=(4, 4), tags="split_line"
            )
        elif preview_canvas:
            (x1, y1) = canvas_points[0]
            x2, y2 = preview_canvas
            self.canvas.create_line(
                x1, y1, x2, y2, fill="yellow", width=2, dash=(4, 4), tags="split_line"
            )

    # ------------------------------------------------------------------ #
    # 疊圖互動
    # ------------------------------------------------------------------ #
    TRANSLATION_KEYS = {
        "q": np.array([0.0, -10.0], dtype=np.float32),
        "w": np.array([0.0, 10.0], dtype=np.float32),
        "e": np.array([-10.0, 0.0], dtype=np.float32),
        "r": np.array([10.0, 0.0], dtype=np.float32),
        "a": np.array([0.0, -1.0], dtype=np.float32),
        "s": np.array([0.0, 1.0], dtype=np.float32),
        "d": np.array([-1.0, 0.0], dtype=np.float32),
        "f": np.array([1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, -0.5], dtype=np.float32),
        "x": np.array([0.0, 0.5], dtype=np.float32),
        "c": np.array([-0.5, 0.0], dtype=np.float32),
        "v": np.array([0.5, 0.0], dtype=np.float32),
    }

    def handle_translation_key(self, key: str):
        if key not in self.TRANSLATION_KEYS:
            return
        shift = self.TRANSLATION_KEYS[key]
        if self.phase == "global_alignment":
            self.global_shift += shift
        elif self.phase == "split_alignment":
            self.right_shift += shift
        self.update_overlay_display()

    def handle_split_line_click(self, canvas_x: int, canvas_y: int):
        if self.roi_rect is None:
            return
        roi_point = self.canvas_to_roi_coordinates(canvas_x, canvas_y)
        if roi_point is None:
            return

        if len(self.split_line_points_roi) == 0:
            self.split_line_points_roi.append(roi_point)
            self.split_line_preview_roi = None
            self.draw_split_line()
            self.update_status("請點選第二個點以完成切割線。")
        elif len(self.split_line_points_roi) == 1:
            self.split_line_points_roi.append(roi_point)
            self.split_line_preview_roi = None
            self.draw_split_line()
            self.enter_split_alignment()

    # ------------------------------------------------------------------ #
    # 狀態更新
    # ------------------------------------------------------------------ #
    def update_status(self, custom: Optional[str] = None):
        if custom:
            self.status_label.config(text=custom)
            return

        if self.phase == "roi_selection":
            msg = "ROI 選取中：拖曳滑鼠框選後按 Enter。"
        elif self.phase == "global_alignment":
            msg = (
                f"全域疊圖：Shift=({self.global_shift[0]:.1f}, {self.global_shift[1]:.1f}) "
                f"對比 {self.global_contrast_current or 0:.4f} / "
                f"{self.global_contrast_best or 0:.4f}。Enter 完成，L 進入切割線模式。"
            )
        elif self.phase == "split_line":
            msg = "切割線模式：點擊兩點建立分割線，Esc 取消。"
        elif self.phase == "split_alignment":
            msg = (
                f"分割疊圖：右側 Shift=({self.right_shift[0]:.1f}, {self.right_shift[1]:.1f}) "
                f"左對比 {self.split_contrast_current_left or 0:.4f} / "
                f"{self.split_contrast_best_left or 0:.4f} | 右對比 {self.split_contrast_current_right or 0:.4f} / "
                f"{self.split_contrast_best_right or 0:.4f}。Enter 完成。"
            )
        elif self.phase == "decision":
            measured = self.latest_measured_mm if self.latest_measured_mm is not None else 0.0
            csv_value = self.latest_csv_mm if self.latest_csv_mm is not None else 0.0
            diff = measured - csv_value
            one_pixel_mm = self._get_one_pixel_mm()
            msg = (
                f"決策：疊圖 {measured:.3f}mm | CSV {csv_value:.3f}mm | 差值 {diff:+.3f}mm "
                f"| 1px≈{one_pixel_mm:.3f}mm。"
                " M=清零、U=更新疊圖值（需完成切割）、K/Enter=保留、B=重選ROI。"
            )
        else:
            msg = ""

        self.status_label.config(text=msg)

    # ------------------------------------------------------------------ #
    # 雜訊判定與儲存
    # ------------------------------------------------------------------ #
    def mark_current_cluster_as_noise(self):
        if self.phase != "decision":
            return
        target = self.suspicious_clusters[self.current_index]
        self._clear_cluster_values(target.cluster_index)
        messagebox.showinfo("已標記", "群集位移已清零。")
        self.export_overlay_image(noise_marked=True)
        self.advance_to_next_cluster()

    def keep_current_cluster(self):
        if self.phase != "decision":
            return
        self.advance_to_next_cluster()

    def calculate_measured_displacement_mm(self) -> float:
        """
        根據當前疊圖結果推估位移（mm）。
        - 分割模式：取右側與左側平移量差異的 Y 分量。
        - 全域模式：直接使用全域平移的 Y 分量。
        """
        if self.roi_pre is None:
            return 0.0

        scale_factor = self.data_manager.scale_factor
        if self.split_mask is None or len(self.split_line_points_roi) != 2:
            # 未進入分割模式，無法可靠推估位移
            return 0.0

        left_shift = self.global_shift
        right_shift = self.global_shift + self.right_shift
        pixel_value = right_shift[1] - left_shift[1]

        displacement_mm = (pixel_value * 10.0) / scale_factor

        cluster = self.data_manager.get_cluster(self.suspicious_clusters[self.current_index].cluster_index)
        orientation = getattr(cluster, "orientation", 0)
        if orientation in (-1, 1):
            displacement_mm = abs(displacement_mm) * orientation

        return displacement_mm

    def calculate_current_csv_displacement(self) -> float:
        """計算目前 CSV 中該群集的位移總和（mm）"""
        target = self.suspicious_clusters[self.current_index]
        cluster = self.data_manager.get_cluster(target.cluster_index)
        values = [
            float(self.data_manager.df.iloc[row_idx, self.data_manager.displacement_col_index])
            for row_idx in range(cluster.start_index, cluster.end_index + 1)
        ]
        return sum(values)

    def apply_overlay_measurement(self):
        """將疊圖測得的位移覆寫至 CSV"""
        if self.phase != "decision":
            return
        if self.latest_measured_mm is None:
            messagebox.showwarning("提示", "尚未取得疊圖位移，請先完成疊圖調整。")
            return
        if self.latest_measured_mm == 0.0:
            messagebox.showwarning("提示", "請先建立切割線並完成第二階段對齊後再更新位移。")
            return

        target = self.suspicious_clusters[self.current_index]
        one_pixel_mm = self._get_one_pixel_mm()
        if abs(self.latest_measured_mm) <= one_pixel_mm:
            self._clear_cluster_values(target.cluster_index)
            messagebox.showinfo(
                "已清零",
                f"疊圖位移 {self.latest_measured_mm:.3f} mm 低於 1 像素閾值 "
                f"({one_pixel_mm:.3f} mm)，已視為雜訊並清零。",
            )
            self.export_overlay_image(noise_marked=True)
            self.advance_to_next_cluster()
            return

        applied = self.data_manager.apply_correction(
            target.cluster_index,
            self.latest_measured_mm,
        )
        if applied:
            messagebox.showinfo("已更新", f"已套用疊圖位移 {self.latest_measured_mm:.3f} mm。")
        else:
            messagebox.showinfo("提醒", "疊圖位移低於閾值，群集已視為雜訊並清零。")
        self.advance_to_next_cluster()

    def export_overlay_image(self, noise_marked: bool):
        if self.current_overlay is None:
            return
        target = self.suspicious_clusters[self.current_index]
        cluster = self.data_manager.get_cluster(target.cluster_index)
        physical = getattr(cluster, "physical_cluster", None)
        if physical is None:
            return

        video_folder = Path("lifts") / "exported_frames" / self.jpg_handler.video_base_name
        video_folder.mkdir(parents=True, exist_ok=True)
        overlay_name = f"static_cluster_{physical.cluster_id:03d}_overlay.png"
        overlay_path = video_folder / overlay_name
        cv2.imwrite(str(overlay_path), self.current_overlay)
        noise_flag = "已清零" if noise_marked else "保留"
        print(f"📸 疊圖快照已儲存：{overlay_path} ({noise_flag})")

    def _clear_cluster_values(self, cluster_index: int):
        """將指定群集在 CSV 中的位移值全部清零"""
        cluster = self.data_manager.get_cluster(cluster_index)
        for row_index in range(cluster.start_index, cluster.end_index + 1):
            self.data_manager.df.iloc[row_index, self.data_manager.displacement_col_index] = 0.0

    def _get_one_pixel_mm(self) -> float:
        """取得相當於一個像素的位移 (mm)"""
        return one_pixel_in_mm(self.data_manager.scale_factor)

    def canvas_to_roi_coordinates(
        self, canvas_x: int, canvas_y: int
    ) -> Optional[Tuple[float, float]]:
        """
        將畫布座標轉換為 ROI 內的像素座標（僅在疊圖階段使用）
        """
        offset_x, offset_y, width, height = self.canvas_bounds
        if not (
            offset_x <= canvas_x <= offset_x + width
            and offset_y <= canvas_y <= offset_y + height
        ):
            return None
        x = (canvas_x - offset_x) / self.display_scale
        y = (canvas_y - offset_y) / self.display_scale
        return float(x), float(y)

    def roi_to_canvas_coordinates(
        self, roi_x: float, roi_y: float
    ) -> Optional[Tuple[int, int]]:
        """
        將 ROI 像素座標轉換為畫布座標
        """
        offset_x, offset_y, width, height = self.canvas_bounds
        if width == 0 or height == 0:
            return None
        canvas_x = int(offset_x + roi_x * self.display_scale)
        canvas_y = int(offset_y + roi_y * self.display_scale)
        return canvas_x, canvas_y

    def save_results_and_exit(self):
        original_path = Path(self.data_manager.csv_path)
        base_name = original_path.name
        if base_name.startswith("mc"):
            new_name = f"mco{base_name[2:]}"
        else:
            new_name = f"mco{base_name}"
        output_path = original_path.parent / new_name
        self.data_manager.df.to_csv(output_path, index=False)
        messagebox.showinfo("已儲存", f"處理完成，檔案已儲存為：\n{output_path}")
        self.root.quit()


def main():
    root = tk.Tk()
    root.withdraw()

    try:
        csv_path_str = filedialog.askopenfilename(
            title="選擇人工校正後的 CSV (mc*.csv)",
            initialdir="lifts/result",
            filetypes=[("CSV 檔案", "*.csv"), ("所有檔案", "*.*")],
        )

        if not csv_path_str:
            return

        csv_path = Path(csv_path_str)
        video_filename = derive_video_filename(csv_path)

        try:
            data_manager = DataManager(str(csv_path), video_filename)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("錯誤", f"資料載入失敗：{exc}")
            return

        suspicious = find_suspicious_clusters(data_manager)
        if not suspicious:
            messagebox.showinfo("提示", "沒有符合條件的疑似雜訊群集。")
            return

        jpg_handler = JPGHandler(video_filename)

        OverlayCleanupApp(
            root=root,
            data_manager=data_manager,
            jpg_handler=jpg_handler,
            suspicious_clusters=suspicious,
        )

        root.mainloop()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()

