"""
Line segment marking workflow for manual measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

Rect = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class MeasurementResult:
    roi: Rect
    samples: List[float]
    average_dy_px: float
    std_dy_px: float


class LineSegmentMarker:
    """
    參考 manual_correction_tool 的流程，提供：
    1. ROI 選取
    2. 左/右畫布線段標記（各 3 次）
    3. 計算 Y 分量變化
    """

    def __init__(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        zoom_factor: int = 3,
    ) -> None:
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.zoom_factor = zoom_factor
        self.roi: Optional[Rect] = None
        self._measurement_points: List[Tuple[Point, Point]] = []

    def perform_measurements(
        self,
        num_measurements: int = 3,
        force_reselect_callback: Optional[Callable[[], None]] = None,
    ) -> Optional[MeasurementResult]:
        if force_reselect_callback:
            force_reselect_callback()
        if not self._select_roi():
            print("⚠️  測量 ROI 選取取消")
            return None

        samples: List[float] = []
        for idx in range(num_measurements):
            print(f"\n📏 第 {idx + 1}/{num_measurements} 次測量 - 左畫布")
            left_line = self._collect_line("Left Measurement", self._get_zoomed_roi(self.left_frame))
            if left_line is None:
                print("⚠️  測量取消")
                return None

            print(f"📏 第 {idx + 1}/{num_measurements} 次測量 - 右畫布")
            right_line = self._collect_line("Right Measurement", self._get_zoomed_roi(self.right_frame))
            if right_line is None:
                print("⚠️  測量取消")
                return None

            left_y = abs(left_line[1][1] - left_line[0][1])
            right_y = abs(right_line[1][1] - right_line[0][1])
            delta = right_y - left_y
            samples.append(delta)
            print(f"  → 左 Y 分量: {left_y:.2f}px, 右 Y 分量: {right_y:.2f}px, Δy={delta:+.2f}px")

        avg = float(np.mean(samples))
        std = float(np.std(samples))
        print(f"\n✅ 測量完成：平均 Δy = {avg:+.2f} ± {std:.2f} 像素")
        return MeasurementResult(roi=self.roi, samples=samples, average_dy_px=avg, std_dy_px=std)

    # ------------------------------------------------------------------ #
    def _select_roi(self) -> bool:
        window_name = "Measurement ROI (Left Frame)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        selection = cv2.selectROI(window_name, self.left_frame, showCrosshair=False)
        cv2.destroyWindow(window_name)
        if selection == (0, 0, 0, 0):
            return False
        x, y, w, h = map(int, selection)
        h_max, w_max = self.left_frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return False
        if x + w > w_max or y + h > h_max:
            return False
        self.roi = (x, y, w, h)
        return True

    def _get_zoomed_roi(self, frame: np.ndarray) -> np.ndarray:
        assert self.roi is not None
        x, y, w, h = self.roi
        roi = frame[y : y + h, x : x + w]
        zoomed = cv2.resize(roi, None, fx=self.zoom_factor, fy=self.zoom_factor, interpolation=cv2.INTER_LINEAR)
        return zoomed

    def _collect_line(self, window_name: str, canvas: np.ndarray) -> Optional[Tuple[Point, Point]]:
        drawing = canvas.copy()
        points: List[Point] = []

        def callback(event, x, y, flags, param):
            nonlocal drawing, points
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(points) == 2:
                    drawing = canvas.copy()
                    points.clear()
                points.append((x, y))
                cv2.circle(drawing, (x, y), 4, (0, 255, 0), -1)
                if len(points) == 2:
                    cv2.line(drawing, points[0], points[1], (0, 255, 0), 2)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, callback)

        while True:
            cv2.imshow(window_name, drawing)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(window_name)
                return None
            if key == ord("r"):
                drawing = canvas.copy()
                points.clear()
            if key in (13, ord(" ")):  # Enter or Space
                if len(points) == 2:
                    break

        cv2.destroyWindow(window_name)
        return self._to_original_coords(points)

    def _to_original_coords(self, points: List[Point]) -> Tuple[Point, Point]:
        assert self.roi is not None
        x, y, _, _ = self.roi
        scale = 1.0 / self.zoom_factor
        converted = []
        for px, py in points:
            original_x = int(x + px * scale)
            original_y = int(y + py * scale)
            converted.append((original_x, original_y))
        return converted[0], converted[1]

