"""
影片路徑與命名相關的 Utilities。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from config import ProjectPaths

VIDEO_GLOB_PATTERNS: Sequence[str] = ("*.mp4", "*.MP4")


@dataclass(frozen=True)
class VideoSource:
    """
    表示一個可供處理的影片來源。
    """

    path: Path
    bucket: str  # e.g. "darkroom_data" or "test_short"
    base_name: str


def get_base_video_name(filename: str) -> str:
    """
    將 darkroom 影片檔名轉換為 base name
    （去掉 a / _a 後綴，以便查詢設定）。
    """

    file_path = Path(filename)
    name = file_path.stem
    suffix = file_path.suffix or ".mp4"

    if name.endswith("_a"):
        name = name[:-2]
    elif name.endswith("a"):
        name = name[:-1]

    return f"{name}{suffix}"


def _scan_directory(directory: Path, bucket: str) -> Iterable[VideoSource]:
    if not directory.exists():
        return []

    sources: List[VideoSource] = []
    seen_paths: set[Path] = set()
    for pattern in VIDEO_GLOB_PATTERNS:
        for video_path in sorted(directory.glob(pattern)):
            if video_path in seen_paths:
                continue
            seen_paths.add(video_path)
            sources.append(
                VideoSource(
                    path=video_path,
                    bucket=bucket,
                    base_name=get_base_video_name(video_path.name),
                )
            )
    return sources


def discover_video_sources(
    paths: ProjectPaths, include_test_clips: bool = False
) -> List[VideoSource]:
    """
    掃描暗房資料夾（及可選的 test clips）以取得可處理影片清單。
    """

    all_sources: List[VideoSource] = []
    all_sources.extend(_scan_directory(paths.darkroom_data_dir, "darkroom_data"))

    if include_test_clips:
        all_sources.extend(_scan_directory(paths.test_video_dir, "test_short"))

    return sorted(all_sources, key=lambda src: (src.bucket, src.path.name.lower()))


def group_sources_by_base(sources: Sequence[VideoSource]) -> Dict[str, List[VideoSource]]:
    """
    依 base name 將影片分組，方便判斷已存在的 a / _a 變體。
    """

    grouped: Dict[str, List[VideoSource]] = {}
    for source in sources:
        grouped.setdefault(source.base_name, []).append(source)
    return grouped

