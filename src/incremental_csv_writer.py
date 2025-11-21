"""
Incremental CSV writer for manual cluster workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from darkroom_csv_schema import SCHEMA


@dataclass
class ClusterCSVRecord:
    cluster_id: int
    start_idx: int
    end_idx: int
    displacement_map: Dict[int, float]
    orientation: int
    pre_frame_path: str
    post_frame_path: str
    fps: float


class IncrementalCSVWriter:
    def __init__(self, csv_path: Path, fps: float, frame_interval: int = 6) -> None:
        self.csv_path = csv_path
        self.fps = fps
        self.frame_interval = frame_interval
        self._load_existing()

    # ------------------------------------------------------------------ #
    def _load_existing(self) -> None:
        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)
            print(f"📂 已載入既有 CSV（{len(self.df)} 筆）")
        else:
            self.df = pd.DataFrame(columns=SCHEMA.columns)

    def save(self) -> None:
        self.df.sort_values("frame_idx", inplace=True)
        self.df.drop_duplicates(subset="frame_idx", keep="last", inplace=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.csv_path, index=False)
        print(f"💾 已儲存 CSV: {self.csv_path}")

    def get_last_processed_frame(self) -> int:
        if self.df.empty:
            return 0
        return int(self.df["frame_idx"].max())

    def get_max_cluster_id(self) -> int:
        if self.df.empty:
            return 0
        return int(self.df["cluster_id"].max())

    def append_cluster(self, record: ClusterCSVRecord) -> None:
        rows: List[Dict[str, object]] = []
        rows.append(
            SCHEMA.normalize_row(
                {
                    "frame_idx": record.start_idx,
                    "second": round(record.start_idx / self.fps, 3),
                    "vertical_travel_distance_mm": 0.0,
                    "cluster_id": 0,
                    "orientation": 0,
                    "frame_path": record.pre_frame_path,
                    "marking_status": "manual",
                }
            )
        )

        for frame_idx, displacement in record.displacement_map.items():
            rows.append(
                SCHEMA.normalize_row(
                    {
                        "frame_idx": frame_idx,
                        "second": round(frame_idx / self.fps, 3),
                        "vertical_travel_distance_mm": displacement,
                        "cluster_id": record.cluster_id,
                        "orientation": record.orientation,
                        "frame_path": "",
                        "marking_status": "manual",
                    }
                )
            )

        rows.append(
            SCHEMA.normalize_row(
                {
                    "frame_idx": record.end_idx,
                    "second": round(record.end_idx / self.fps, 3),
                    "vertical_travel_distance_mm": 0.0,
                    "cluster_id": 0,
                    "orientation": 0,
                    "frame_path": record.post_frame_path,
                    "marking_status": "manual",
                }
            )
        )

        new_df = pd.DataFrame(rows)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.save()

    def delete_cluster(self, cluster_id: int) -> pd.DataFrame:
        mask_cluster = self.df["cluster_id"] == cluster_id
        mask_refs = self.df["frame_path"].str.contains(
            f"cluster_{cluster_id:03d}.jpg", na=False
        )
        mask = mask_cluster | mask_refs
        deleted_rows = self.df.loc[mask].copy()
        if deleted_rows.empty:
            print(f"ℹ️  CSV 中找不到 cluster {cluster_id} 的紀錄")
            return deleted_rows
        self.df = self.df.loc[~mask]
        print(f"🗑️  CSV 中移除 cluster {cluster_id} 的 {len(deleted_rows)} 筆記錄")
        self.save()
        return deleted_rows

    def restore_rows(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        self.df = pd.concat([self.df, rows], ignore_index=True)
        print(f"↩️  已恢復 {len(rows)} 筆 CSV 記錄")
        self.save()

