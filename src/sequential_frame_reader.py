"""
逐幀讀取器 SequentialFrameReader（對應重構計畫的階段 B）。

特點：
- 完全依序讀取，不使用 ``VideoCapture.set`` 做隨機跳轉
- 支援向後導航時重新開啟影片並從頭爬行
- 維護過去/未來幀快取（供後續 GUI/標記流程使用）
- 對外提供 read_next_keyframe / seek_to_frame / get_frame_at_offset 等 API
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


class SequentialFrameReader:
    """
    嚴格遵守「只做順序讀取」原則的影片讀取器。
    """

    def __init__(
        self,
        video_path: str | Path,
        frame_interval: int = 6,
        backward_cache_size: int = 1600,
        forward_cache_size: int = 400,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"找不到影片：{self.video_path}")

        self.frame_interval = frame_interval
        self.backward_cache_size = backward_cache_size
        self.forward_cache_size = forward_cache_size

        self.vidcap = self._open_capture()
        self.video_length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.vidcap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 else 30.0

        self.next_frame_index = 0  # 下一次 read() 會得到的幀索引
        self._current_frame_idx: Optional[int] = None
        self._current_frame: Optional[np.ndarray] = None

        self.backward_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.forward_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # 公開 API
    # ------------------------------------------------------------------ #
    @property
    def last_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        if self._current_frame_idx is None or self._current_frame is None:
            return None, None
        return self._current_frame_idx, self._current_frame.copy()

    def get_current_frame(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        return self.last_frame

    def read_next_keyframe(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """讀取下一個 frame_interval 的倍數幀。"""

        target_idx = self._next_keyframe_index(self.next_frame_index)
        if target_idx >= self.video_length:
            return None, None
        frame = self.seek_to_frame(target_idx)
        if frame is None:
            return None, None
        return self._current_frame_idx, frame

    def seek_to_frame(self, target_idx: int) -> Optional[np.ndarray]:
        """
        依序讀取直到目標幀，返回該幀影像。

        Args:
            target_idx: 目標幀（必須是 frame_interval 的倍數）
        """

        aligned_idx = self._align_to_interval(target_idx)
        if aligned_idx != target_idx:
            raise ValueError(
                f"target_idx 必須為 {self.frame_interval} 的倍數：{target_idx}"
            )
        if aligned_idx < 0 or aligned_idx >= self.video_length:
            return None

        if aligned_idx < self._last_read_index():
            self._reset_capture()

        frame = self._crawl_to_frame(aligned_idx)
        return frame

    def get_frame_at_offset(self, base_idx: int, offset: int) -> Optional[np.ndarray]:
        """
        從 base_idx 讀取偏移 offset 的幀（結果同樣會與 frame_interval 對齊）。
        """

        target_idx = base_idx + offset
        target_idx = self._align_to_interval(target_idx)
        if target_idx < 0 or target_idx >= self.video_length:
            return None
        return self.seek_to_frame(target_idx)

    def reset(self) -> None:
        """重置讀取位置到影片開頭。"""

        self._reset_capture()

    def close(self) -> None:
        if self.vidcap is not None:
            self.vidcap.release()
            self.vidcap = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # 內部工具
    # ------------------------------------------------------------------ #
    def _open_capture(self) -> cv2.VideoCapture:
        vidcap = cv2.VideoCapture(str(self.video_path))
        if not vidcap.isOpened():
            raise RuntimeError(f"無法開啟影片：{self.video_path}")
        return vidcap

    def _reset_capture(self) -> None:
        self.close()
        self.vidcap = self._open_capture()
        self.next_frame_index = 0
        self.backward_cache.clear()
        self.forward_cache.clear()
        self._current_frame_idx = None
        self._current_frame = None

    def _read_frame_raw(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        if self.vidcap is None:
            return None, None
        ret, frame = self.vidcap.read()
        if not ret or frame is None:
            return None, None
        frame_idx = self.next_frame_index
        self.next_frame_index += 1
        return frame_idx, frame

    def _crawl_to_frame(self, target_idx: int) -> Optional[np.ndarray]:
        while self.next_frame_index < target_idx:
            idx, frame = self._read_frame_raw()
            if idx is None or frame is None:
                return None
            if idx % self.frame_interval == 0:
                self._add_to_cache(self.backward_cache, idx, frame, self.backward_cache_size)

        idx, frame = self._read_frame_raw()
        if idx is None or frame is None:
            return None

        self._current_frame_idx = idx
        self._current_frame = frame.copy()
        self._add_to_cache(self.backward_cache, idx, frame, self.backward_cache_size)
        return self._current_frame.copy()

    def _add_to_cache(
        self,
        cache: "OrderedDict[int, np.ndarray]",
        idx: int,
        frame: np.ndarray,
        max_size: int,
    ) -> None:
        cache[idx] = frame.copy()
        cache.move_to_end(idx)
        if len(cache) > max_size:
            cache.popitem(last=False)

    def _align_to_interval(self, idx: int) -> int:
        return (idx // self.frame_interval) * self.frame_interval

    def _next_keyframe_index(self, idx: int) -> int:
        remainder = idx % self.frame_interval
        if remainder == 0:
            return idx
        return idx + (self.frame_interval - remainder)

    def _last_read_index(self) -> int:
        if self.next_frame_index == 0:
            return -1
        return self.next_frame_index - 1

    # ------------------------------------------------------------------ #
    # 釋放資源
    # ------------------------------------------------------------------ #
    def __del__(self) -> None:
        self.close()

