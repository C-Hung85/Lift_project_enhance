#!/usr/bin/env python3
"""
簡易幀匯出工具

用途：從 lifts/data/1.mp4 匯出指定範圍的幀為 PNG 到專案根目錄下的 investigation 目錄。

預設：
- 影片：lifts/data/1.mp4
- 幀範圍：6844 ~ 6882（含）
- 旋轉：依據 src/rotation_config.py 自動套用（如有設定）

可參數化：
- --video <path>
- --start <int>
- --end <int>
- --output <dir>
- --no-rotate  關閉旋轉
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Optional

import cv2


def rotate_if_needed(frame, video_name: str, apply_rotation: bool):
    if not apply_rotation:
        return frame
    # 嘗試從同目錄載入 rotation_config 與工具
    try:
        from rotation_config import rotation_config  # type: ignore
        from rotation_utils import rotate_frame  # type: ignore
    except Exception:
        return frame

    angle = rotation_config.get(video_name, 0)
    if not angle:
        return frame
    return rotate_frame(frame, angle)


def read_frame_exact(cap: cv2.VideoCapture, target_index: int) -> Optional[cv2.Mat]:
    """嘗試嚴格定位到 target_index 並讀取當前幀，最多嘗試 3 次。"""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        pos_err = abs(actual_pos - target_index)
        if pos_err <= 1:
            ret, frame = cap.read()
            if ret:
                return frame
        # 重新開啟再試
        cap.release()
        cap.open(cap.getBackendName())  # 這行可能無法重開，改以外層控制
    return None


def export_frames(
    video_path: Path,
    output_dir: Path,
    start_index: int,
    end_index: int,
    apply_rotation: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"❌ 找不到影片檔案: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 無法開啟影片檔案: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vname = video_path.name
    print(f"影片: {video_path}")
    print(f"FPS: {fps:.6f}, 總幀數: {total}")
    print(f"輸出目錄: {output_dir}")
    print(f"匯出範圍: {start_index} ~ {end_index} (含)")
    print(f"旋轉: {'開啟' if apply_rotation else '關閉'}")

    # 逐幀匯出
    for idx in range(start_index, end_index + 1):
        if idx < 0 or idx >= total:
            print(f"跳過 {idx}: 超出範圍 (0 ~ {total-1})")
            continue

        # 簡化但穩健：set → 讀取；若失敗則再嘗試一次
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        actual_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret or frame is None:
            # 重試一次
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"❌ 讀取幀失敗: {idx} (實際定位: {actual_pos})")
                continue

        # 旋轉（如需）
        frame = rotate_if_needed(frame, vname, apply_rotation)

        out_path = output_dir / f"frame_{idx}.png"
        ok = cv2.imwrite(str(out_path), frame)
        if ok:
            print(f"✅ 存檔: {out_path}")
        else:
            print(f"❌ 存檔失敗: {out_path}")

    cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="匯出影片指定幀為 PNG")
    parser.add_argument("--video", type=str, default=str(Path("lifts") / "data" / "1.mp4"))
    parser.add_argument("--start", type=int, default=6844)
    parser.add_argument("--end", type=int, default=6882)
    parser.add_argument("--output", type=str, default=str(Path("investigation")))
    parser.add_argument("--no-rotate", action="store_true", help="不套用 rotation_config 旋轉")
    return parser.parse_args()


def main():
    args = parse_args()

    # 將輸出目錄解析為專案根目錄下的路徑
    # 若本檔位於 src/，其上一層應為專案根
    project_root = Path(__file__).resolve().parents[1]

    video_path = (project_root / args.video).resolve()
    output_dir = (project_root / args.output).resolve()

    export_frames(
        video_path=video_path,
        output_dir=output_dir,
        start_index=args.start,
        end_index=args.end,
        apply_rotation=not args.no_rotate,
    )


if __name__ == "__main__":
    main()


