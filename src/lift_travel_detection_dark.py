"""
新暗房標註系統的入口點（階段 A：環境準備與資源盤點）。

此版本僅負責：
1. 載入整體設定（路徑、時間窗口、旋轉配置等）
2. 掃描可處理的影片清單並列出摘要
3. 顯示比例尺快取 / 暗房區間 / CSV schema 等資訊

後續階段會在這個骨架上逐步加入 SequentialFrameReader、
OpenCVGUIPlayer、Cluster 工作流與 CSV 寫入等模組。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

from config import (
    DarkroomProjectConfig,
    VideoTimeWindow,
    load_darkroom_project_config,
)
from darkroom_csv_schema import SCHEMA
from darkroom_utils import get_darkroom_intervals_for_video, print_darkroom_summary
from darkroom_video_utils import (
    VideoSource,
    discover_video_sources,
    group_sources_by_base,
    get_base_video_name,
)
from scale_cache_utils import (
    get_missing_videos,
    load_scale_cache,
    print_cache_status,
)
from sequential_frame_reader import SequentialFrameReader
from opencv_gui_player import OpenCVGUIPlayer

try:
    from rotation_config import rotation_config
except ImportError:
    rotation_config = {}

try:
    from darkroom_intervals import darkroom_intervals
except ImportError:
    darkroom_intervals = {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="暗房標註系統（重構版）- 階段 A 設定檢視工具"
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="列出 lifts/test_short 內的測試片段",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default=(),
        help="依檔名關鍵字過濾（可多個）",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="顯示比例尺快取及暗房區間摘要",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="印出 CSV schema 後結束",
    )
    parser.add_argument(
        "--probe-video",
        help="指定影片檔名或路徑，使用 SequentialFrameReader 進行測試",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=5,
        help="probe 模式：連續讀取的 keyframe 數量",
    )
    parser.add_argument(
        "--probe-offsets",
        type=int,
        nargs="*",
        default=None,
        help="probe 模式：從最後一個 keyframe 進行 offset 尋址（單位：幀）",
    )
    parser.add_argument(
        "--run-gui",
        help="指定影片檔名或路徑，啟動 OpenCV GUI 播放器",
    )
    return parser.parse_args(argv)


def format_window(window: VideoTimeWindow | None) -> str:
    if not window:
        return "-"

    start = window.start_sec if window.start_sec is not None else 0
    end = window.end_sec if window.end_sec is not None else float("inf")
    start_str = f"{start:>6.1f}s"
    end_str = "∞" if end == float("inf") else f"{end:>6.1f}s"
    return f"{start_str} ~ {end_str}"


def format_roi(window: VideoTimeWindow | None) -> str:
    if not window or window.roi_ratio is None:
        return "-"
    return f"{window.roi_ratio:.2f}"


def format_darkroom_info(video_name: str) -> str:
    intervals, has = get_darkroom_intervals_for_video(video_name, darkroom_intervals)
    if not has:
        return "-"
    return f"{len(intervals)} intervals"


def format_rotation_info(video_name: str) -> str:
    if video_name not in rotation_config:
        return "-"
    angle = rotation_config[video_name]
    return f"{angle:+.1f}°"


def format_scale_info(video_name: str, scale_config: dict[str, float]) -> str:
    scale_value = scale_config.get(video_name)
    if scale_value is None:
        return "⚠ missing"
    return f"{scale_value:.2f} px/10mm"


def apply_filters(
    sources: List[VideoSource], filters: Sequence[str]
) -> List[VideoSource]:
    if not filters:
        return sources

    lowered = [token.lower() for token in filters]
    filtered: List[VideoSource] = []
    for source in sources:
        name_lower = source.path.name.lower()
        if any(token in name_lower for token in lowered):
            filtered.append(source)
    return filtered


def print_table(
    sources: Sequence[VideoSource],
    config: DarkroomProjectConfig,
    scale_config: dict[str, float],
) -> None:
    headers = (
        "Video File",
        "Bucket",
        "Base",
        "Start ~ End (s)",
        "ROI",
        "Darkroom",
        "Rotation",
        "Scale Cache",
    )
    rows: List[Sequence[str]] = []

    for source in sources:
        window = config.video_windows.get(source.base_name)
        rows.append(
            (
                source.path.name,
                source.bucket,
                source.base_name,
                format_window(window),
                format_roi(window),
                format_darkroom_info(source.base_name),
                format_rotation_info(source.base_name),
                format_scale_info(source.base_name, scale_config),
            )
        )

    if not rows:
        print("⚠️  沒有符合條件的影片")
        return

    col_widths = [
        max(len(str(value)) for value in [header, *[row[i] for row in rows]])
        for i, header in enumerate(headers)
    ]

    def _print_row(values: Sequence[str]) -> None:
        line = " | ".join(str(value).ljust(col_widths[idx]) for idx, value in enumerate(values))
        print(line)

    separator = "-+-".join("-" * width for width in col_widths)

    _print_row(headers)
    print(separator)
    for row in rows:
        _print_row(row)


def summarize_sources(
    sources: Sequence[VideoSource],
    config: DarkroomProjectConfig,
    scale_config: dict[str, float],
) -> None:
    total = len(sources)
    grouped = group_sources_by_base(sources)
    missing_windows = [
        base for base in grouped if base not in config.video_windows
    ]
    missing_scale = [
        base for base in grouped if base not in scale_config
    ]
    missing_rotation = [
        base for base in grouped if base not in rotation_config
    ]

    print("\n📊 Summary")
    print(f"- Total video files listed : {total}")
    print(f"- Unique base names        : {len(grouped)}")
    print(f"- Missing time window cfg  : {len(missing_windows)}")
    print(f"- Missing scale cache      : {len(missing_scale)}")
    print(f"- Missing rotation override: {len(missing_rotation)}")

    if missing_windows:
        print(f"  ⚠ 未設定時間窗口：{', '.join(sorted(missing_windows))}")
    if missing_scale:
        print(f"  ⚠ 缺少比例尺：{', '.join(sorted(missing_scale))}")
    if missing_rotation:
        print(f"  ℹ 未設定旋轉角：{', '.join(sorted(missing_rotation))}")


def resolve_video_path(config: DarkroomProjectConfig, raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.exists():
        return candidate

    fallback = config.paths.darkroom_data_dir / raw_value
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"找不到指定影片：{raw_value}（嘗試路徑：{candidate} / {fallback})"
    )


def run_probe_session(
    video_path: Path,
    count: int,
    offsets: Sequence[int],
) -> None:
    print("\n=== SequentialFrameReader Probe ===")
    print(f"Video     : {video_path}")
    reader = SequentialFrameReader(video_path)
    duration = reader.video_length / reader.fps if reader.fps else 0.0
    print(
        f"Frames/FPS: {reader.video_length} / {reader.fps:.3f} "
        f"(≈ {duration:.2f}s)"
    )

    last_idx = None
    for i in range(count):
        frame_idx, _ = reader.read_next_keyframe()
        if frame_idx is None:
            print("  ⏹ 已達影片結尾")
            break
        last_idx = frame_idx
        seconds = frame_idx / reader.fps if reader.fps else 0.0
        print(f"  ▶ Keyframe #{i+1:02d} -> frame {frame_idx} ({seconds:.3f}s)")

    if last_idx is not None and offsets:
        print(f"\n  ↪ Offset lookup from base frame {last_idx}:")
        for offset in offsets:
            target = last_idx + offset
            frame = reader.get_frame_at_offset(last_idx, offset)
            if frame is None:
                print(f"    offset {offset:+}: 超出範圍或無法讀取 ({target})")
                continue
            idx, _ = reader.get_current_frame()
            seconds = (idx or 0) / reader.fps if reader.fps else 0.0
            print(f"    offset {offset:+}: frame {idx} ({seconds:.3f}s)")
        reader.seek_to_frame(last_idx)

    reader.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.show_schema:
        SCHEMA.describe()
        return

    project_config = load_darkroom_project_config()
    project_config.paths.ensure_base_directories()

    scale_config, scale_cache_info = load_scale_cache()

    print("=== 暗房標註系統：資源盤點（階段 A）===")
    print(f"Data root : {project_config.paths.data_root}")
    print(f"Scan step : {project_config.scan.frame_interval} frames")

    sources = discover_video_sources(
        project_config.paths, include_test_clips=args.include_tests
    )

    sources = apply_filters(sources, args.filter)
    print_table(sources, project_config, scale_config)
    summarize_sources(sources, project_config, scale_config)

    if args.show_diagnostics:
        print("\n=== 比例尺快取 ===")
        if scale_cache_info:
            print(f"版本資訊: {scale_cache_info}")
        expected_videos = [source.base_name for source in sources]
        missing_for_list = get_missing_videos(scale_config, expected_videos)
        print_cache_status(scale_config, missing_for_list)

        print("\n=== 暗房時間區間 ===")
        print_darkroom_summary(darkroom_intervals)

    if args.probe_video:
        probe_offsets = args.probe_offsets if args.probe_offsets is not None else [60, -60]
        try:
            video_path = resolve_video_path(project_config, args.probe_video)
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
        else:
            run_probe_session(video_path, max(1, args.probe_count), probe_offsets)

    if args.run_gui:
        try:
            video_path = resolve_video_path(project_config, args.run_gui)
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
            return
        base_name = get_base_video_name(video_path.name)
        scale_factor = scale_config.get(base_name)
        player = OpenCVGUIPlayer(str(video_path), scale_factor=scale_factor)
        player.run()


if __name__ == "__main__":
    main()

