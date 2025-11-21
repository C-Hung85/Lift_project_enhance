"""
ClusterManager: 協調 CSV 與 JPG 刪除，確保資料一致。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from incremental_csv_writer import IncrementalCSVWriter


class ClusterManager:
    def __init__(self, csv_writer: IncrementalCSVWriter, export_dir: Path) -> None:
        self.csv_writer = csv_writer
        self.export_dir = export_dir
        self._undo_stack: list[dict] = []

    def delete_cluster(self, cluster_id: int) -> bool:
        jpg_snapshots = self._snapshot_jpg(cluster_id)
        deleted_rows = self.csv_writer.delete_cluster(cluster_id)
        if deleted_rows.empty and not jpg_snapshots:
            print("ℹ️  沒有找到可刪除的資料，略過")
            return False
        self._undo_stack.append(
            {
                "cluster_id": cluster_id,
                "rows": deleted_rows,
                "jpgs": jpg_snapshots,
            }
        )
        print(f"↩️  刪除紀錄加入 undo stack（大小 {len(self._undo_stack)}）")
        return True

    def undo_last_delete(self) -> Optional[int]:
        if not self._undo_stack:
            print("ℹ️  沒有可復原的刪除")
            return None
        snapshot = self._undo_stack.pop()
        cluster_id = snapshot["cluster_id"]
        for path, content in snapshot["jpgs"]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                print(f"↩️  已還原 {path}")
            except OSError as exc:
                print(f"⚠️  還原 {path} 失敗: {exc}")
        rows = snapshot["rows"]
        if rows is not None and not rows.empty:
            self.csv_writer.restore_rows(rows)
        print(f"✅ 已復原 cluster {cluster_id}")
        return cluster_id

    def _snapshot_jpg(self, cluster_id: int):
        snapshots = []
        patterns = [
            f"pre_cluster_{cluster_id:03d}.jpg",
            f"post_cluster_{cluster_id:03d}.jpg",
        ]
        for filename in patterns:
            path = self.export_dir / filename
            if path.exists():
                try:
                    snapshots.append((path, path.read_bytes()))
                except OSError as exc:
                    print(f"⚠️  讀取 {path} 失敗: {exc}")
        self._delete_jpg(cluster_id)
        return snapshots

    def _delete_jpg(self, cluster_id: int) -> None:
        patterns = [
            f"pre_cluster_{cluster_id:03d}.jpg",
            f"post_cluster_{cluster_id:03d}.jpg",
        ]
        for filename in patterns:
            path = self.export_dir / filename
            if path.exists():
                try:
                    path.unlink()
                    print(f"🗑️  已刪除 {path}")
                except OSError as exc:
                    print(f"⚠️  刪除 {path} 失敗: {exc}")
            else:
                print(f"ℹ️  找不到 {path}，略過")

