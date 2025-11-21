from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

__all__ = [
    "Config",
    "video_config",
    "ProjectPaths",
    "ScanSettings",
    "VideoTimeWindow",
    "DarkroomProjectConfig",
    "load_darkroom_project_config",
    "get_video_window",
]


@dataclass(frozen=True)
class ProjectPaths:
    """
    儲存與暗房標註流程相關的常用路徑。
    """

    repo_root: Path
    data_root: Path
    lifts_dir: Path
    darkroom_data_dir: Path
    test_video_dir: Path
    exported_frames_dir: Path
    result_dir: Path
    scale_images_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ProjectPaths":
        env = env or os.environ
        repo_root = Path(__file__).resolve().parents[1]
        data_root = Path(env.get("SATA", repo_root))
        lifts_dir = data_root / "lifts"
        return cls(
            repo_root=repo_root,
            data_root=data_root,
            lifts_dir=lifts_dir,
            darkroom_data_dir=lifts_dir / "darkroom_data",
            test_video_dir=lifts_dir / "test_short",
            exported_frames_dir=lifts_dir / "exported_frames",
            result_dir=lifts_dir / "result",
            scale_images_dir=lifts_dir / "scale_images",
        )

    def ensure_base_directories(self) -> None:
        """
        建立標註流程所需的基本資料夾。
        """

        for path in {
            self.lifts_dir,
            self.darkroom_data_dir,
            self.test_video_dir,
            self.exported_frames_dir,
            self.result_dir,
            self.scale_images_dir,
        }:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScanSettings:
    frame_interval: int = 6


@dataclass(frozen=True)
class VideoTimeWindow:
    """
    定義單支影片的處理時間範圍與 ROI 參數（單位：秒）。
    """

    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    roi_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        data: Dict[str, float] = {}
        if self.start_sec is not None:
            data["start"] = float(self.start_sec)
        if self.end_sec is not None:
            data["end"] = float(self.end_sec)
        if self.roi_ratio is not None:
            data["roi_ratio"] = float(self.roi_ratio)
        return data


@dataclass(frozen=True)
class DarkroomProjectConfig:
    paths: ProjectPaths
    scan: ScanSettings
    video_windows: Dict[str, VideoTimeWindow]


def _window(
    start: Optional[float] = None,
    end: Optional[float] = None,
    roi_ratio: Optional[float] = None,
) -> VideoTimeWindow:
    return VideoTimeWindow(start_sec=start, end_sec=end, roi_ratio=roi_ratio)


VIDEO_WINDOWS: Dict[str, VideoTimeWindow] = {
    "2.mp4": _window(start=110),
    "4-2.mp4": _window(end=532),
    "5-2.mp4": _window(start=23, end=396),
    "8-2.mp4": _window(end=960),
    "9.mp4": _window(start=32, end=1334),
    "6.mp4": _window(end=618),
    "11.mp4": _window(end=293),
    "12.mp4": _window(start=152, end=1181),
    "7.mp4": _window(end=1367),
    "10.mp4": _window(start=20),
    "13.mp4": _window(end=801),
    "14.mp4": _window(start=19, end=196),
    "15.mp4": _window(start=262, end=789),
    "17.mp4": _window(start=25),
    "18.mp4": _window(start=18, end=257),
    "16.mp4": _window(start=284),
    "19.mp4": _window(end=610),
    "21.mp4": _window(start=30, end=798),
    "22.mp4": _window(start=35, end=650),
    "23.mp4": _window(start=30, end=660),
    "26.mp4": _window(start=30, end=660),
    "28.mp4": _window(start=36, end=15 * 60 + 55),
    "29.mp4": _window(start=40, end=12 * 60 + 30),
    "30.mp4": _window(start=30, end=15 * 60),
    "31-1.mp4": _window(start=10, end=12 * 60 + 16),
    "31-2.mp4": _window(start=28 * 60),
    "34.mp4": _window(end=3 * 60 + 36),
    "36-1.mp4": _window(end=2 * 60 + 24),
    "36-2.mp4": _window(start=7 * 60 + 4),
    "37.mp4": _window(start=10),
    "38-1.mp4": _window(start=16, end=8 * 60),
    "38-2.mp4": _window(start=8 * 60 + 55, end=13 * 60 + 15),
    "39.mp4": _window(start=7, end=16 * 60 + 15),
    "40.mp4": _window(end=6 * 60 + 55),
    "41.mp4": _window(end=18 * 60 + 30, roi_ratio=0.7),
    "42.mp4": _window(end=14 * 60),
    "43.mp4": _window(start=60, end=12 * 60),
    "45.mp4": _window(start=5, end=12 * 60 + 15),
    "46.mp4": _window(start=5, end=13 * 60 + 2),
}


def load_darkroom_project_config(
    env: Mapping[str, str] | None = None,
) -> DarkroomProjectConfig:
    """
    建立完整的暗房專案設定 dataclass。
    """

    paths = ProjectPaths.from_env(env)
    scan = ScanSettings()
    return DarkroomProjectConfig(paths=paths, scan=scan, video_windows=VIDEO_WINDOWS)


def get_video_window(video_name: str) -> Optional[VideoTimeWindow]:
    """
    提供特定影片的時間窗口設定。
    """

    return VIDEO_WINDOWS.get(video_name)


PROJECT_CONFIG = load_darkroom_project_config()
PROJECT_CONFIG.paths.ensure_base_directories()

Config: Dict[str, MutableMapping[str, object]] = {
    "files": {"data_folder": str(PROJECT_CONFIG.paths.data_root)},
    "scan_setting": {"interval": PROJECT_CONFIG.scan.frame_interval},
}

video_config: Dict[str, Dict[str, float]] = {
    name: window.to_dict() for name, window in PROJECT_CONFIG.video_windows.items()
}