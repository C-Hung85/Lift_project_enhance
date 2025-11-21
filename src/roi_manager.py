"""
Playback ROI manager with in-canvas selection workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]


@dataclass
class ROIManager:
    frame_width: int
    frame_height: int
    min_size: int = 40
    zoom_factor: float = 2.0

    def __post_init__(self) -> None:
        self._playback_roi: Optional[Rect] = None
        self._selecting: bool = False
        self._selection_mode: Optional[str] = None
        self._anchor: Optional[Point] = None
        self._current: Optional[Point] = None

    # ---- 基本狀態 ----
    def has_playback_roi(self) -> bool:
        return self._playback_roi is not None

    def clear_playback_roi(self) -> None:
        self._playback_roi = None

    def begin_selection(self, mode: str = "playback") -> None:
        """
        啟用 ROI 選取模式（於 Dual Canvas 內拖曳）。
        """

        self._selecting = True
        self._selection_mode = mode
        self._anchor = None
        self._current = None

    def cancel_selection(self) -> None:
        self._selecting = False
        self._selection_mode = None
        self._anchor = None
        self._current = None

    def is_selecting(self) -> bool:
        return self._selecting

    # ---- 滑鼠事件 ----
    def handle_mouse_down(self, x: int, y: int) -> None:
        if not self._selecting:
            return
        self._anchor = self._clamp_point(x, y)
        self._current = self._anchor

    def handle_mouse_move(self, x: int, y: int) -> None:
        if not self._selecting or self._anchor is None:
            return
        self._current = self._clamp_point(x, y)

    def handle_mouse_up(self) -> bool:
        if not self._selecting or self._anchor is None or self._current is None:
            self.cancel_selection()
            return False

        rect = self._rect_from_points(self._anchor, self._current)
        if rect[2] < self.min_size or rect[3] < self.min_size:
            self.cancel_selection()
            return False

        if self._selection_mode == "playback":
            self._playback_roi = rect

        self.cancel_selection()
        return True

    # ---- 畫面處理 ----
    def draw_selection_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._selecting or self._anchor is None or self._current is None:
            return frame

        output = frame.copy()
        x1, y1 = self._anchor
        x2, y2 = self._current
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 200, 255),
            2,
        )
        cv2.putText(
            output,
            "Selecting ROI - release to confirm / press R to cancel",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
        )
        return output

    def apply_playback_roi(self, frame: np.ndarray) -> np.ndarray:
        if not self._playback_roi:
            return frame

        x, y, w, h = self._playback_roi
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return frame
        resized = cv2.resize(
            roi,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def set_playback_roi(self, roi: Rect | None) -> None:
        self._playback_roi = roi

    # ---- 工具方法 ----
    def _clamp_point(self, x: int, y: int) -> Point:
        return (
            max(0, min(self.frame_width - 1, x)),
            max(0, min(self.frame_height - 1, y)),
        )

    @staticmethod
    def _rect_from_points(p1: Point, p2: Point) -> Rect:
        x1, y1 = p1
        x2, y2 = p2
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        return (x_min, y_min, width, height)

