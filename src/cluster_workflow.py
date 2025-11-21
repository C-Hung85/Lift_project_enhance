"""
Cluster workflow helper for manual marking stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from frame_exporter import DEFAULT_EXPORT_ROOT, export_frame_jpg


@dataclass
class ActiveCluster:
    cluster_id: int
    start_idx: int
    pre_frame_path: Path


@dataclass
class CompletedCluster:
    cluster_id: int
    start_idx: int
    end_idx: int
    pre_frame_path: Path
    post_frame_path: Path


class ClusterWorkflow:
    """
    控管 cluster 建立 / 結束 / 取消。
    """

    def __init__(self, video_name: str, export_root: Path | None = None) -> None:
        self.video_name = video_name
        self.export_root = export_root or DEFAULT_EXPORT_ROOT
        self._cluster_counter = 0
        self._active: Optional[ActiveCluster] = None

    # ------------------------------------------------------------------ #
    def has_active_cluster(self) -> bool:
        return self._active is not None

    def get_active_cluster(self) -> Optional[ActiveCluster]:
        return self._active

    def start_cluster(self, frame_idx: int, frame: np.ndarray) -> ActiveCluster:
        if self._active is not None:
            raise RuntimeError("已有尚未完成的 cluster")
        self._cluster_counter += 1
        cluster_id = self._cluster_counter
        pre_name = f"pre_cluster_{cluster_id:03d}.jpg"
        pre_path = export_frame_jpg(
            self.video_name,
            frame_idx,
            frame,
            pre_name,
            self.export_root,
        )
        self._active = ActiveCluster(
            cluster_id=cluster_id,
            start_idx=frame_idx,
            pre_frame_path=pre_path,
        )
        return self._active

    def complete_cluster(self, frame_idx: int, frame: np.ndarray) -> CompletedCluster:
        if self._active is None:
            raise RuntimeError("沒有待完成的 cluster")

        post_name = f"post_cluster_{self._active.cluster_id:03d}.jpg"
        post_path = export_frame_jpg(
            self.video_name,
            frame_idx,
            frame,
            post_name,
            self.export_root,
        )
        completed = CompletedCluster(
            cluster_id=self._active.cluster_id,
            start_idx=self._active.start_idx,
            end_idx=frame_idx,
            pre_frame_path=self._active.pre_frame_path,
            post_frame_path=post_path,
        )
        self._active = None
        return completed

    def cancel_active(self) -> None:
        if self._active is None:
            return
        try:
            self._active.pre_frame_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._active = None

