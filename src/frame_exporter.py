"""
Utility helpers for exporting frames to JPG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_EXPORT_ROOT = Path("lifts") / "exported_frames"


def export_frame_jpg(
    video_name: str,
    frame_idx: int,
    frame: np.ndarray,
    filename: str,
    export_root: Path | None = None,
) -> Path:
    """
    將單張影像輸出為 JPG 檔案。
    """

    root = export_root or DEFAULT_EXPORT_ROOT
    export_dir = root / f"{video_name}_dark"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / filename
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    success = cv2.imwrite(str(export_path), frame, encode_param)
    if not success:
        raise RuntimeError(f"無法匯出 JPG：{export_path}")
    return export_path

