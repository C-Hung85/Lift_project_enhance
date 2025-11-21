"""
暗房標註流程的 CSV 結構定義。

此模組集中管理欄位名稱、預設值與型別對應，避免由程式碼片段
各自硬編字串。為後續的 IncrementalCSVWriter / ClusterManager 等元件
提供單一資訊來源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Tuple

CSV_COLUMNS: Tuple[str, ...] = (
    "frame_idx",
    "second",
    "vertical_travel_distance_mm",
    "cluster_id",
    "orientation",
    "frame_path",
    "marking_status",
)

CSV_DTYPE: Mapping[str, type] = {
    "frame_idx": int,
    "second": float,
    "vertical_travel_distance_mm": float,
    "cluster_id": int,
    "orientation": int,
    "frame_path": str,
    "marking_status": str,
}

DEFAULT_ROW_VALUES: Mapping[str, object] = {
    "frame_idx": 0,
    "second": 0.0,
    "vertical_travel_distance_mm": 0.0,
    "cluster_id": 0,
    "orientation": 0,
    "frame_path": "",
    "marking_status": "manual",
}


@dataclass(frozen=True)
class DarkroomCsvSchema:
    columns: Tuple[str, ...] = CSV_COLUMNS
    dtypes: Mapping[str, type] = field(default_factory=lambda: CSV_DTYPE)
    defaults: Mapping[str, object] = field(default_factory=lambda: DEFAULT_ROW_VALUES)

    def empty_row(self) -> Dict[str, object]:
        """
        產生一筆符合 schema 的預設資料列。
        """

        return {column: self.defaults.get(column) for column in self.columns}

    def normalize_row(self, row: MutableMapping[str, object]) -> Dict[str, object]:
        """
        依 schema 欄位填補缺值；額外欄位會被忽略。
        """

        normalized = {}
        for column in self.columns:
            normalized[column] = row.get(column, self.defaults.get(column))
        return normalized

    def describe(self) -> None:
        """
        將 schema 摘要輸出至 console（便於檢查）。
        """

        print("📄 暗房 CSV Schema")
        print("-" * 40)
        for column in self.columns:
            dtype = self.dtypes.get(column, object)
            default = self.defaults.get(column, "(無預設)")
            print(f"{column:30s} type={dtype.__name__:<6s} default={default}")
        print("-" * 40)


SCHEMA = DarkroomCsvSchema()

