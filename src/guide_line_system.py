"""
GuideLineSystem: 控制雙畫布上的水平輔助線顯示/拖曳。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class GuideLineSystem:
    frame_height: int
    color: Tuple[int, int, int] = (0, 255, 255)
    thickness: int = 2
    adjustment_color: Tuple[int, int, int] = (0, 255, 255)

    def __post_init__(self) -> None:
        self.visible: bool = True
        self.adjustment_mode: bool = False
        self.y_position: int = self.frame_height // 2
        self.dragging: bool = False

    # ---- 公開 API -----------------------------------------------------
    def toggle_visibility(self) -> None:
        self.visible = not self.visible

    def toggle_adjustment_mode(self) -> None:
        self.adjustment_mode = not self.adjustment_mode
        if self.adjustment_mode:
            self.visible = True

    def start_dragging(self, mouse_y: int) -> None:
        if not self.adjustment_mode:
            return
        if abs(mouse_y - self.y_position) <= 20:
            self.dragging = True

    def update_position(self, mouse_y: int) -> None:
        if not self.dragging:
            return
        self.y_position = max(0, min(self.frame_height - 1, mouse_y))

    def stop_dragging(self) -> None:
        self.dragging = False

    def draw_on_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.visible:
            return frame
        output = frame.copy()
        h, w = output.shape[:2]
        color = self.adjustment_color if self.adjustment_mode else self.color
        thickness = self.thickness + (1 if self.adjustment_mode else 0)

        cv2.line(
            output,
            (0, self.y_position),
            (w - 1, self.y_position),
            color,
            thickness,
        )

        marker_size = 12
        cv2.line(
            output,
            (0, max(0, self.y_position - marker_size)),
            (0, min(h - 1, self.y_position + marker_size)),
            color,
            thickness,
        )
        cv2.line(
            output,
            (w - 1, max(0, self.y_position - marker_size)),
            (w - 1, min(h - 1, self.y_position + marker_size)),
            color,
            thickness,
        )

        if self.adjustment_mode:
            cv2.putText(
                output,
                "Guide Line Adjustment Mode - Drag to adjust",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return output

    def serialize_state(self) -> Tuple[bool, bool, int]:

        return (self.visible, self.adjustment_mode, self.y_position)

    def restore_state(self, visible: bool, adjustment: bool, y_position: int) -> None:
        self.visible = visible
        self.adjustment_mode = adjustment
        self.y_position = max(0, min(self.frame_height - 1, y_position))

