"""
OpenCVGUIPlayer: 雙畫布播放核心（對應重構階段 C）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    tk = None
    messagebox = None

from guide_line_system import GuideLineSystem
from roi_manager import ROIManager
from sequential_frame_reader import SequentialFrameReader
from cluster_workflow import ClusterWorkflow, CompletedCluster
from line_segment_marker import LineSegmentMarker, MeasurementResult
from incremental_csv_writer import IncrementalCSVWriter, ClusterCSVRecord
from cluster_manager import ClusterManager


@dataclass
class PlayerState:
    playing: bool = False
    clahe_enabled: bool = True
    right_offset: int = 60
    playback_delay_ms: int = 150


class OpenCVGUIPlayer:
    """
    提供雙畫布 + 控制面板的基本播放體驗。
    """

    def __init__(
        self,
        video_path: str,
        right_offset: int = 60,
        scale_factor: Optional[float] = None,
    ) -> None:
        self.video_path = video_path
        self.reader = SequentialFrameReader(video_path)
        idx, frame = self.reader.read_next_keyframe()
        if idx is None or frame is None:
            raise RuntimeError("無法讀取影片開頭關鍵幀")

        self.current_frame_idx = idx
        self.current_frame = frame
        self.frame_height, self.frame_width = frame.shape[:2]
        self.state = PlayerState(right_offset=right_offset)
        self.roi_manager = ROIManager(self.frame_width, self.frame_height)
        self.guide_line_system = GuideLineSystem(frame_height=self.frame_height)
        self.video_name = Path(video_path).stem
        self.export_root = Path("lifts") / "exported_frames"
        self.cluster_workflow = ClusterWorkflow(self.video_name, self.export_root)
        self.cluster_locked_frame_idx: Optional[int] = None
        self.cluster_locked_frame: Optional[np.ndarray] = None
        csv_path = (
            Path("lifts") / "result" / f"{self.video_name}_dark.csv"
        )
        fps = self.reader.fps if self.reader.fps else 30.0
        self.csv_writer = IncrementalCSVWriter(csv_path, fps, self.reader.frame_interval)
        export_dir = self.export_root / f"{self.video_name}_dark"
        self.cluster_manager = ClusterManager(self.csv_writer, export_dir)
        self.last_completed_cluster_id: Optional[int] = None
        self.scale_factor_px_per_10mm = scale_factor
        self._resume_from_csv()

        self.buttons = self._build_button_layout()
        self.mouse_pos_control = (0, 0)
        self.mouse_pos_canvas = (0, 0)
        self.last_mouse_update = time.time()
        self.mouse_throttle = 0.05

        cv2.namedWindow("Dual Canvas", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Dual Canvas", self._mouse_callback_canvas)
        cv2.setMouseCallback("Control Panel", self._mouse_callback_control)

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ #
    def _resume_from_csv(self) -> None:
        last_frame = self.csv_writer.get_last_processed_frame()
        last_cluster = self.csv_writer.get_max_cluster_id()
        if last_cluster > 0:
            self.last_completed_cluster_id = last_cluster

        if last_frame <= self.current_frame_idx:
            return

        print(f"⏩ 載入既有進度，前往 frame {last_frame}")
        frame = self.reader.seek_to_frame(last_frame)
        if frame is None:
            print("⚠️  無法定位到既有進度，維持初始位置")
            return
        self.current_frame_idx = last_frame
        self.current_frame = frame

    # ------------------------------------------------------------------ #
    def _build_button_layout(self) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
        return {
            "play": {"rect": (10, 10, 110, 60), "label": "[Play]"},
            "pause": {"rect": (120, 10, 220, 60), "label": "[Pause]"},
            "forward_6": {"rect": (230, 10, 310, 60), "label": "[+6]"},
            "backward_6": {"rect": (320, 10, 400, 60), "label": "[-6]"},
            "forward_30": {"rect": (410, 10, 510, 60), "label": "[+30]"},
            "backward_30": {"rect": (520, 10, 620, 60), "label": "[-30]"},
            "forward_300": {"rect": (630, 10, 750, 60), "label": "[+300]"},
            "backward_300": {"rect": (760, 10, 890, 60), "label": "[-300]"},
            "toggle_clahe": {"rect": (10, 70, 170, 120), "label": "[CLAHE]"},
            "mark_roi": {"rect": (180, 70, 340, 120), "label": "[Set ROI]"},
            "clear_roi": {"rect": (350, 70, 510, 120), "label": "[Clear ROI]"},
            "guide_toggle": {"rect": (520, 70, 660, 120), "label": "[Guide G]"},
            "guide_show": {"rect": (670, 70, 820, 120), "label": "[Guide H]"},
            "cluster_start": {"rect": (10, 130, 200, 180), "label": "[Mark Start]"},
            "cluster_end": {"rect": (210, 130, 400, 180), "label": "[Mark End]"},
            "cluster_cancel": {"rect": (410, 130, 600, 180), "label": "[Cancel Cluster]"},
            "cluster_delete": {"rect": (610, 130, 780, 180), "label": "[Delete Last]"},
            "cluster_undo_delete": {"rect": (10, 185, 210, 230), "label": "[Undo Delete]"},
        }

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("▶ OpenCV GUI Player 啟動，按 Q 鍵離開。")
        try:
            while True:
                dual_frame = self._prepare_dual_canvas()
                control_panel = self._draw_control_panel()
                cv2.imshow("Dual Canvas", dual_frame)
                cv2.imshow("Control Panel", control_panel)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if self._handle_keypress(key):
                        break

                if self.state.playing:
                    self._advance_frame()
                    delay = max(10, self.state.playback_delay_ms)
                    time.sleep(delay / 1000.0)
        finally:
            cv2.destroyWindow("Dual Canvas")
            cv2.destroyWindow("Control Panel")
            self.reader.close()

    # ------------------------------------------------------------------ #
    def _prepare_dual_canvas(self) -> np.ndarray:
        if self.cluster_locked_frame is not None and self.cluster_locked_frame_idx is not None:
            left_source = self.cluster_locked_frame
            left_idx = self.cluster_locked_frame_idx
        else:
            left_source = self.current_frame
            left_idx = self.current_frame_idx

        left_frame = left_source.copy()
        right_frame = self._get_right_frame(self.current_frame_idx)

        left_frame = self._apply_clahe_if_needed(left_frame)
        right_frame = self._apply_clahe_if_needed(right_frame)

        selecting_roi = self.roi_manager.is_selecting()
        if not selecting_roi and self.roi_manager.has_playback_roi():
            left_frame = self.roi_manager.apply_playback_roi(left_frame)
            right_frame = self.roi_manager.apply_playback_roi(right_frame)

        left_frame = self.roi_manager.draw_selection_overlay(left_frame)

        left_frame = self.guide_line_system.draw_on_frame(left_frame)
        right_frame = self.guide_line_system.draw_on_frame(right_frame)

        left_frame = self._draw_status_text(left_frame, left_idx, "Left")
        if self.cluster_workflow.has_active_cluster():
            left_frame = self._draw_cluster_badge(left_frame, left_idx)
        right_idx = self.current_frame_idx + self.state.right_offset
        right_frame = self._draw_status_text(right_frame, right_idx, "Right")

        dual_frame = np.hstack([left_frame, right_frame])
        return dual_frame

    def _draw_cluster_badge(self, frame: np.ndarray, start_idx: int) -> np.ndarray:
        output = frame.copy()
        active = self.cluster_workflow.get_active_cluster()
        if not active:
            return output
        label = f"Cluster #{active.cluster_id:03d} | Start {start_idx}"
        cv2.rectangle(output, (10, 100), (400, 140), (30, 30, 30), -1)
        cv2.putText(
            output,
            label,
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return output

    def _apply_clahe_if_needed(self, frame: np.ndarray) -> np.ndarray:
        if not self.state.clahe_enabled:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _draw_status_text(self, frame: np.ndarray, frame_idx: int, label: str) -> np.ndarray:
        output = frame.copy()
        seconds = frame_idx / self.reader.fps if self.reader.fps else 0.0
        cv2.putText(
            output,
            f"{label}: Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            f"Time: {seconds:.2f}s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            output,
            f"CLAHE: {'ON' if self.state.clahe_enabled else 'OFF'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if self.state.clahe_enabled else (100, 100, 100),
            2,
        )
        return output

    def _get_right_frame(self, base_idx: int) -> np.ndarray:
        offset_idx = base_idx + self.state.right_offset
        frame = self.reader.get_frame_at_offset(base_idx, self.state.right_offset)
        if frame is None:
            frame = np.zeros_like(self.current_frame)
        self.reader.seek_to_frame(base_idx)
        return frame

    # ------------------------------------------------------------------ #
    def _draw_control_panel(self) -> np.ndarray:
        panel = np.zeros((240, 900, 3), dtype=np.uint8)
        panel[:] = (55, 55, 55)

        for btn_name, btn_info in self.buttons.items():
            x1, y1, x2, y2 = btn_info["rect"]
            label = btn_info["label"]
            mx, my = self.mouse_pos_control
            is_hover = x1 <= mx <= x2 and y1 <= my <= y2
            color = (90, 200, 90) if is_hover else (80, 80, 80)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(panel, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return panel

    # ------------------------------------------------------------------ #
    def _mouse_callback_canvas(self, event, x, y, flags, param) -> None:
        now = time.time()
        if self.roi_manager.is_selecting():
            if x >= self.frame_width:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_manager.handle_mouse_down(x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if now - self.last_mouse_update < self.mouse_throttle:
                    return
                self.last_mouse_update = now
                self.roi_manager.handle_mouse_move(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                success = self.roi_manager.handle_mouse_up()
                if success:
                    print("📐 播放 ROI 已設定完成")
                else:
                    print("⚠️  播放 ROI 選取取消或範圍過小")
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if now - self.last_mouse_update < self.mouse_throttle:
                return
            self.last_mouse_update = now
            self.mouse_pos_canvas = (x, y)
            self.guide_line_system.update_position(y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.guide_line_system.start_dragging(y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.guide_line_system.stop_dragging()

    def _mouse_callback_control(self, event, x, y, flags, param) -> None:
        now = time.time()
        if event == cv2.EVENT_MOUSEMOVE:
            if now - self.last_mouse_update < self.mouse_throttle:
                return
            self.last_mouse_update = now
            self.mouse_pos_control = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            for btn_name, btn in self.buttons.items():
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._on_button_click(btn_name)
                    break

    # ------------------------------------------------------------------ #
    def _on_button_click(self, btn_name: str) -> None:
        actions = {
            "play": lambda: self._set_playing(True),
            "pause": lambda: self._set_playing(False),
            "forward_6": lambda: self._jump_frames(6),
            "backward_6": lambda: self._jump_frames(-6),
            "forward_30": lambda: self._jump_frames(30),
            "backward_30": lambda: self._jump_frames(-30),
            "forward_300": lambda: self._jump_frames(300),
            "backward_300": lambda: self._jump_frames(-300),
            "toggle_clahe": self._toggle_clahe,
            "mark_roi": self._set_roi,
            "clear_roi": self._clear_roi,
            "guide_toggle": self._toggle_guide_mode,
            "guide_show": self.guide_line_system.toggle_visibility,
            "cluster_start": self._start_cluster,
            "cluster_end": self._complete_cluster,
            "cluster_cancel": self._cancel_cluster,
            "cluster_delete": self._delete_last_cluster,
            "cluster_undo_delete": self._undo_last_delete,
        }
        action = actions.get(btn_name)
        if action:
            action()

    # ------------------------------------------------------------------ #
    def _handle_keypress(self, key: int) -> bool:
        if key == ord("q"):
            return True
        if key == ord(" "):
            self.state.playing = not self.state.playing
        elif key == ord("c"):
            self._toggle_clahe()
        elif key == ord("g"):
            self._toggle_guide_mode()
        elif key == ord("h"):
            self.guide_line_system.toggle_visibility()
        elif key == ord("r"):
            self._set_roi()
        elif key == ord("R"):
            self._clear_roi()
        elif key == ord("f"):
            self._jump_frames(6)
        elif key == ord("b"):
            self._jump_frames(-6)
        elif key == 13:  # Enter
            if self.cluster_workflow.has_active_cluster():
                self._complete_cluster()
            else:
                self._start_cluster()
        elif key == ord("d"):
            self._cancel_cluster()
        elif key == ord("u"):
            self._undo_last_delete()
        elif key == 27 and self.roi_manager.is_selecting():
            self.roi_manager.cancel_selection()
            print("❌ 已取消 ROI 選取")
        return False

    def _set_playing(self, playing: bool) -> None:
        self.state.playing = playing and not self.guide_line_system.adjustment_mode

    def _toggle_clahe(self) -> None:
        self.state.clahe_enabled = not self.state.clahe_enabled

    def _toggle_guide_mode(self) -> None:
        self.guide_line_system.toggle_adjustment_mode()
        if self.guide_line_system.adjustment_mode:
            self.state.playing = False

    def _set_roi(self) -> None:
        if self.roi_manager.is_selecting():
            self.roi_manager.cancel_selection()
            print("❌ 已取消 ROI 選取")
            return
        self.state.playing = False
        self.roi_manager.begin_selection()
        print("✏️  在左畫布拖曳選擇播放 ROI，放開滑鼠確認")

    def _clear_roi(self) -> None:
        self.roi_manager.clear_playback_roi()
        self.roi_manager.cancel_selection()
        print("🧹 已清除播放 ROI")

    # ------------------------------------------------------------------ #
    def _advance_frame(self) -> None:
        next_idx, next_frame = self.reader.read_next_keyframe()
        if next_idx is None or next_frame is None:
            print("⏹ 已達影片結尾")
            self.state.playing = False
            return
        self.current_frame_idx = next_idx
        self.current_frame = next_frame

    def _jump_frames(self, delta: int) -> None:
        target = self.current_frame_idx + delta
        target = max(0, min(target, self.reader.video_length - 1))
        target = (target // self.reader.frame_interval) * self.reader.frame_interval
        frame = self.reader.seek_to_frame(target)
        if frame is None:
            print("⚠️  無法跳轉到指定幀")
            return
        self.current_frame_idx = target
        self.current_frame = frame

    def _start_cluster(self) -> None:
        if self.cluster_workflow.has_active_cluster():
            print("⚠️  已有啟動中的 Cluster")
            return
        try:
            active = self.cluster_workflow.start_cluster(
                self.current_frame_idx, self.current_frame.copy()
            )
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return
        self.cluster_locked_frame_idx = active.start_idx
        self.cluster_locked_frame = self.current_frame.copy()
        self.state.playing = False
        print(f"🔒 Cluster #{active.cluster_id:03d} 已開始 (frame {active.start_idx})")

    def _complete_cluster(self) -> None:
        if not self.cluster_workflow.has_active_cluster():
            print("⚠️  沒有啟動中的 Cluster")
            return
        try:
            completed = self.cluster_workflow.complete_cluster(
                self.current_frame_idx, self.current_frame.copy()
            )
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return
        print(
            f"✅ Cluster #{completed.cluster_id:03d} 完成 "
            f"({completed.start_idx} → {completed.end_idx})"
        )
        measurement = self._run_measurement_workflow()
        if measurement:
            print(
                f"📊 測量結果: Δy={measurement.average_dy_px:+.2f} ± {measurement.std_dy_px:.2f} px"
            )
            record = self._build_csv_record(completed, measurement)
            if record:
                self.csv_writer.append_cluster(record)
                self.last_completed_cluster_id = record.cluster_id
        else:
            print("⚠️  使用者取消測量流程")
        self.cluster_locked_frame = None
        self.cluster_locked_frame_idx = None

    def _cancel_cluster(self) -> None:
        if not self.cluster_workflow.has_active_cluster():
            print("ℹ️  目前沒有 Cluster 可取消")
            return
        self.cluster_workflow.cancel_active()
        self.cluster_locked_frame = None
        self.cluster_locked_frame_idx = None
        print("🗑️  已取消 Cluster 並刪除 pre 圖片")

    def _undo_last_delete(self) -> None:
        restored = self.cluster_manager.undo_last_delete()
        if restored is not None:
            self.last_completed_cluster_id = restored

    def _delete_last_cluster(self) -> None:
        if self.last_completed_cluster_id is None:
            print("ℹ️  尚未有可刪除的 cluster")
            return
        success = self.cluster_manager.delete_cluster(self.last_completed_cluster_id)
        if success:
            print(f"🗑️  已刪除 cluster {self.last_completed_cluster_id}")
            self.last_completed_cluster_id = None

    def _run_measurement_workflow(self):
        if self.cluster_locked_frame is None:
            print("⚠️  找不到 Cluster pre frame，無法進行測量")
            return None

        def force_reselect():
            self.roi_manager.clear_playback_roi()
            print("↩️  進入測量模式，播放 ROI 已回到全幅")

        marker = LineSegmentMarker(
            self.cluster_locked_frame,
            self.current_frame,
        )
        measurement = marker.perform_measurements(force_reselect_callback=force_reselect)
        if measurement and self._prompt_apply_measurement_roi(marker):
            self.roi_manager.set_playback_roi(marker.roi)
            print("📐 已將測量 ROI 套用到播放模式")
        return measurement

    def _prompt_apply_measurement_roi(self, marker: LineSegmentMarker) -> bool:
        if marker.roi is None:
            return False
        prompt = (
            "是否將此次測量 ROI 套用至播放模式？\n"
            "Yes：沿用測量視窗以利後續瀏覽\n"
            "No：維持原播放 ROI 設定"
        )
        if messagebox and tk:
            root = tk.Tk()
            root.withdraw()
            try:
                result = messagebox.askyesno("套用測量 ROI", prompt)
            finally:
                root.destroy()
            return result
        else:
            resp = input(f"{prompt} [y/N]: ")
            return resp.strip().lower().startswith("y")

    def _confirm_orientation(self, avg_px: float, total_mm: float, orientation: int) -> int:
        direction_text = "UP" if orientation == 1 else "DOWN"
        message = (
            "方向判定\n"
            f"Δy = {avg_px:+.2f} px\n"
            f"位移 = {total_mm:.2f} mm {direction_text}\n\n"
            "若方向正確請選 Yes，若需要反轉請選 No。"
        )
        if messagebox and tk:
            root = tk.Tk()
            root.withdraw()
            try:
                confirmed = messagebox.askyesno("確認方向", message)
            finally:
                root.destroy()
            return orientation if confirmed else -orientation
        else:
            resp = input(f"{message}\nConfirm direction? [Y/n]: ")
            return orientation if resp.strip().lower() in ("", "y", "yes") else -orientation

    def _build_csv_record(
        self,
        completed: CompletedCluster,
        measurement: MeasurementResult,
    ) -> Optional[ClusterCSVRecord]:
        if not self.scale_factor_px_per_10mm or self.scale_factor_px_per_10mm == 0:
            print("⚠️  缺少比例尺資料，無法寫入 CSV")
            return None
        avg_px = measurement.average_dy_px
        if avg_px == 0:
            print("⚠️  Δy 為 0，跳過寫入 CSV")
            return None
        orientation = 1 if avg_px > 0 else -1
        total_mm = abs(avg_px) * 10 / self.scale_factor_px_per_10mm
        orientation = self._confirm_orientation(avg_px, total_mm, orientation)
        signed_total_mm = total_mm * orientation
        displacement_map = self._distribute_displacement_uniformly(
            completed.start_idx,
            completed.end_idx,
            signed_total_mm,
        )
        if not displacement_map:
            print("⚠️  起訖幀之間沒有可用關鍵幀，跳過寫入 CSV")
            return None
        record = ClusterCSVRecord(
            cluster_id=completed.cluster_id,
            start_idx=completed.start_idx,
            end_idx=completed.end_idx,
            displacement_map=displacement_map,
            orientation=orientation,
            pre_frame_path=completed.pre_frame_path.name,
            post_frame_path=completed.post_frame_path.name,
            fps=self.reader.fps if self.reader.fps else 30.0,
        )
        return record

    def _distribute_displacement_uniformly(
        self,
        start_idx: int,
        end_idx: int,
        total_mm: float,
    ) -> Dict[int, float]:
        interval = self.reader.frame_interval
        start_key = ((start_idx // interval) + 1) * interval
        end_key = ((end_idx // interval) - 1) * interval
        if end_key < start_key:
            return {}
        keyframes = list(range(start_key, end_key + 1, interval))
        if not keyframes:
            return {}
        per_frame = total_mm / len(keyframes)
        return {idx: per_frame for idx in keyframes}

    # ------------------------------------------------------------------ #
    def __del__(self) -> None:
        self.reader.close()

