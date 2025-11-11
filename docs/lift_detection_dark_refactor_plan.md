# 暗房電梯運動偵測系統重構計畫書

## 1. 專案背景與問題分析

### 1.1 現狀問題
- 現有自動偵測演算法在暗房環境下無法正常運作
- CLAHE + ORB 特徵匹配在低光環境下成效不佳
- 需要可靠的人工標註系統取代自動偵測

### 1.2 解決方案
開發**半自動電腦輔助人工標註系統**，結合：
- 可靠的逐幀讀取機制（避免 OpenCV 幀導航誤差）
- 直覺的雙畫布影片播放器
- 精確的線段標記法（參考 manual_correction_tool.py）
- 增量式 CSV 輸出（支援中斷與繼續）

---

## 2. 核心設計變更

### 2.1 移除的功能
- ❌ 特徵偵測與匹配（ORB + BF Matcher）
- ❌ 自動運動候選判定
- ❌ 狀態機處理（Idle/PendingEnter/InCluster）
- ❌ 物理群集自動偵測
- ❌ 正負抖動幀對過濾
- ❌ Camera pan 偵測
- ❌ 自動化運動距離計算
- ❌ Inspection 影片生成

### 2.2 保留的功能
- ✅ **逐幀讀取邏輯**（順序讀取，避免 `vidcap.set()` 的幀導航誤差）
- ✅ **比例尺快取系統**（scale_cache_utils + 紅點偵測）
- ✅ **影片旋轉支援**（rotation_config + rotate_frame）
- ✅ **暗房區間設定**（darkroom_intervals）
- ✅ **JPG 匯出功能**（pre/post cluster frames）
- ✅ **CSV 輸出功能**（結構調整，支援增量寫入）
- ✅ **CLAHE 前處理**（作為可開關的顯示增強選項）

### 2.3 新增的功能
- 🆕 **雙畫布影片播放器**（OpenCV 原生 GUI，高效能）
- 🆕 **時間軸導航控制**（前進/後退/暫停/變速）
- 🆕 **Cluster 標記工作流**（起始幀 → 結束幀 → 線段標記）
- 🆕 **三次線段標記取平均**（像素位移 → mm 轉換）
- 🆕 **等速分配演算法**（位移平均分配至關鍵幀）
- 🆕 **增量式 CSV 寫入**（即時儲存，支援中斷繼續）
- 🆕 **自動方向判定**（根據線段標記計算，提示使用者確認）
- 🆕 **標記錯誤處理**（清除當前線段重繪、刪除整個 Cluster）
- 🆕 **快捷鍵系統**（暫停、儲存、復原等關鍵操作）
- 🆕 **輔助線系統**（可調整位置的水平參考線，橫跨左右畫布）

### 2.4 技術棧選擇說明
**主界面 GUI**：
- ✅ 100% OpenCV 原生 GUI（`cv2.imshow()` + 滑鼠回調 + 鍵盤快捷鍵）
- 理由：高效能、快速響應、低系統負載

**確認對話框**：
- ✅ 使用 `tkinter.messagebox`（標準庫）
- 理由：對話框只在標記過程中使用（已暫停），不影響主界面響應速度
- 優先考慮：開發效率、可靠性、使用者熟悉的系統原生外觀

**設計原則**：效能關鍵路徑使用 OpenCV，非關鍵部分優先考慮開發效率與可靠性

---

## 3. 系統架構設計

### 3.1 模組結構

```
lift_travel_detection_dark.py (重構後)
│
├─ [保留] 比例尺快取載入模組
│   ├─ load_scale_cache()
│   ├─ 紅點偵測與距離計算
│   └─ video_scale_dict 建立
│
├─ [保留] 影片前處理模組
│   ├─ get_base_video_name()
│   ├─ preprocess_darkroom_frame() (CLAHE - 可開關)
│   └─ rotate_frame() 整合
│
├─ [新增] 逐幀讀取器類別 (SequentialFrameReader)
│   ├─ __init__(video_path, frame_interval=6)
│   ├─ read_next_frame() → (frame_idx, frame)
│   ├─ seek_to_frame(target_idx) → 從當前位置順序爬行
│   ├─ get_frame_at_offset(offset) → 順序讀取偏移幀
│   └─ 內部維護當前讀取位置與幀快取
│
├─ [新增] 雙畫布播放器 GUI (OpenCVGUIPlayer)
│   ├─ 使用 cv2.imshow() 和 cv2.namedWindow() 實作
│   ├─ 雙畫布設計：單一視窗並排顯示左右幀（3840x1080）
│   ├─ 左幀：當前幀 (frame_idx)
│   ├─ 右幀：可自訂對照間隔（預設 +60 幀）
│   ├─ 播放控制：使用滑鼠點擊區域實現按鈕
│   ├─ 微調控制：±6/±30/±300 幀（使用順序讀取）
│   ├─ 快捷鍵系統：Space(暫停), S(儲存), Z(復原), H(輔助線)
│   ├─ 滑鼠回調：節流更新頻率防止崩潰（兩個回調：雙畫布、控制面板）
│   ├─ 輔助線系統：可調整的水平參考線（黃色，橫跨整個視窗）
│   └─ 狀態顯示：直接繪製文字於影像上
│
├─ [新增] Cluster 標記工作流 (ClusterMarkingWorkflow)
│   ├─ mark_cluster_start() → cluster_id
│   ├─ lock_left_canvas()
│   ├─ navigate_right_canvas_independently()
│   ├─ mark_cluster_end() → 觸發 JPG 匯出
│   ├─ start_line_marking() → 進入標記模式
│   ├─ auto_determine_orientation() → 自動計算方向
│   ├─ show_confirmation_dialog() → 提示使用者確認
│   └─ delete_cluster() → 刪除錯誤標記
│
├─ [新增] 線段標記模組 (LineSegmentMarker)
│   ├─ 參考 manual_correction_tool.py 實作
│   ├─ mark_line_on_canvas(canvas_side, roi_zoom=8)
│   ├─ repeat_3_times_and_average()
│   ├─ clear_current_marking() → 清除當前線段重繪
│   ├─ calculate_pixel_displacement()
│   └─ convert_to_mm(scale_factor)
│
├─ [新增] 等速分配演算法 (UniformDistributor)
│   ├─ distribute_displacement(start_idx, end_idx, total_mm)
│   ├─ 計算關鍵幀數量：(end_idx - start_idx) / 6
│   └─ 每幀分配：total_mm / num_keyframes
│
├─ [新增] 增量 CSV 管理器 (IncrementalCSVWriter)
│   ├─ initialize_csv(video_name) → 建立或載入 CSV
│   ├─ append_cluster(cluster_data) → 即時寫入
│   ├─ delete_cluster(cluster_id) → 刪除 CSV 記錄（僅 CSV）
│   ├─ load_existing_progress() → 支援中斷繼續
│   ├─ get_last_processed_frame() → 取得上次處理位置
│   └─ CSV 結構：frame_idx, second, displacement_mm, cluster_id, frame_path

├─ [新增] Cluster 管理器 (ClusterManager)
│   ├─ delete_cluster(cluster_id) → 協調 CSV 與 JPG 刪除
│   ├─ 確保資料一致性（先查詢 JPG 路徑 → 刪除 JPG → 刪除 CSV）
│   ├─ 錯誤處理與使用者回饋
│   └─ 整合 IncrementalCSVWriter 與檔案系統操作
│
└─ [保留] JPG 匯出模組
    ├─ export_frame_jpg(frame_data, jpg_filename, video_name)
    └─ 路徑：lifts/exported_frames/<video_name>_dark/
```

---

## 4. 關鍵技術設計

### 4.1 逐幀讀取器 (SequentialFrameReader)

**設計目標：完全避免 OpenCV `vidcap.set()` 的幀導航誤差**

```python
class SequentialFrameReader:
    """
    順序讀取影片幀，避免使用 OpenCV 的隨機存取功能

    重要：所有幀導航都透過順序讀取實現，包括：
    - 載入已有 CSV 檔案時的初始定位
    - 所有跳轉操作（+6, +30, +300, -6, -30, -300）
    - 右畫布的對照幀讀取
    """

    def __init__(self, video_path, frame_interval=6):
        self.vidcap = cv2.VideoCapture(video_path)
        self.frame_interval = frame_interval
        self.current_position = 0  # 實際讀取位置
        self.backward_cache = {}  # 過去幀快取（1600 幀）
        self.forward_cache = {}   # 未來幀快取（400 幀）
        self.backward_cache_size = 1600  # 80% 容量，涵蓋過去約 2.7 分鐘
        self.forward_cache_size = 400    # 20% 容量，涵蓋未來約 40 秒
        self.video_length = int(self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_next_keyframe(self):
        """
        讀取下一個關鍵幀（6的倍數）
        從當前位置順序讀取直到下一個關鍵幀
        """
        # 跳過非關鍵幀
        while self.current_position % self.frame_interval != 0:
            ret, _ = self.vidcap.read()
            if not ret:
                return None, None
            self.current_position += 1

        # 讀取關鍵幀
        ret, frame = self.vidcap.read()
        if ret:
            self.frame_cache[self.current_position] = frame.copy()
            self._maintain_cache()
            result_idx = self.current_position
            self.current_position += 1
            return result_idx, frame
        return None, None

    def seek_to_frame(self, target_idx):
        """
        從當前位置順序讀取到目標幀

        Args:
            target_idx: 目標幀索引（必須是 6 的倍數）

        Returns:
            frame or None
        """
        if target_idx % self.frame_interval != 0:
            raise ValueError(f"target_idx 必須是 {self.frame_interval} 的倍數")

        # 檢查快取（優先檢查 backward_cache，其次 forward_cache）
        if target_idx in self.backward_cache:
            return self.backward_cache[target_idx]
        if target_idx in self.forward_cache:
            return self.forward_cache[target_idx]

        # 判斷方向
        if target_idx < self.current_position:
            # 向後導航：需要重新開啟影片並從頭讀取
            print(f"⚠️ 向後導航 {self.current_position} → {target_idx}，重新開啟影片")
            self.vidcap.release()
            self.vidcap = cv2.VideoCapture(self.video_path)
            self.current_position = 0
            self.backward_cache.clear()
            self.forward_cache.clear()

        # 順序讀取到目標幀
        print(f"📖 順序讀取幀 {self.current_position} → {target_idx}")
        while self.current_position < target_idx:
            ret, frame = self.vidcap.read()
            if not ret:
                return None

            # 快取關鍵幀到 backward_cache
            if self.current_position % self.frame_interval == 0:
                self.backward_cache[self.current_position] = frame.copy()
                self._maintain_cache()

            self.current_position += 1

        # 讀取目標幀
        ret, frame = self.vidcap.read()
        if ret:
            self.backward_cache[self.current_position] = frame.copy()
            self._maintain_cache()
            self.current_position += 1
            return frame
        return None

    def get_frame_at_offset(self, base_idx, offset):
        """
        從 base_idx 讀取偏移 offset 幀（支援正向/反向）

        注意：這是語法糖（syntactic sugar），內部直接呼叫 seek_to_frame。
        提供此方法是為了提高程式碼可讀性，避免手動計算關鍵幀對齊。

        Args:
            base_idx: 基準幀索引
            offset: 偏移量（可正可負，但結果必須是 6 的倍數）

        Returns:
            frame or None
        """
        target_idx = base_idx + offset

        # 確保目標是關鍵幀（自動對齊到 6 的倍數）
        target_idx = (target_idx // self.frame_interval) * self.frame_interval

        if target_idx < 0 or target_idx >= self.video_length:
            return None

        # 直接呼叫 seek_to_frame，避免重複邏輯
        return self.seek_to_frame(target_idx)

    def _maintain_cache(self):
        """維護快取大小（雙向快取：80% 過去，20% 未來）"""
        # 維護 backward_cache
        if len(self.backward_cache) > self.backward_cache_size:
            oldest_idx = min(self.backward_cache.keys())
            del self.backward_cache[oldest_idx]

        # 維護 forward_cache
        if len(self.forward_cache) > self.forward_cache_size:
            oldest_idx = min(self.forward_cache.keys())
            del self.forward_cache[oldest_idx]

    def reset(self):
        """重置讀取器到影片開頭"""
        self.vidcap.release()
        self.vidcap = cv2.VideoCapture(self.video_path)
        self.current_position = 0
        self.backward_cache.clear()
        self.forward_cache.clear()
```

**關鍵設計決策：**
- ✅ 完全避免 `vidcap.set(cv2.CAP_PROP_POS_FRAMES, target)`
- ✅ 所有導航操作都使用順序讀取
- ✅ 向後導航時重新開啟影片從頭讀取
- ✅ 維護幀快取減少重複讀取
- ✅ 支援載入已有 CSV 時的初始定位

---

### 4.2 OpenCV 原生 GUI 播放器

**設計理念：使用 OpenCV 實現高效能 GUI**

為了獲得更快的響應速度並避免 Tkinter 的效能瓶頸，採用 OpenCV 原生的視窗和滑鼠回調系統。

**技術實作：**

```python
class OpenCVGUIPlayer:
    """
    使用 OpenCV 原生 GUI 實作的雙畫布播放器

    特色：
    - 使用 cv2.imshow() 顯示影像
    - 單一視窗並排顯示左右幀（3840x1080）
    - 直接在影像上繪製文字和按鈕
    - 滑鼠回調實現點擊區域檢測
    - 鍵盤快捷鍵快速響應
    - 滑鼠座標更新節流（防止系統崩潰）
    """

    def __init__(self, video_path, scale_factor):
        self.frame_reader = SequentialFrameReader(video_path)
        self.scale_factor = scale_factor
        self.frame_width = 1920   # 單幀寬度
        self.frame_height = 1080  # 單幀高度

        # GUI 狀態
        self.playing = False
        self.clahe_enabled = True
        self.right_offset = 60
        self.mouse_pos_control = (0, 0)  # 控制面板滑鼠位置
        self.mouse_pos_canvas = (0, 0)   # 雙畫布滑鼠位置
        self.last_mouse_update = time.time()
        self.mouse_throttle = 0.05  # 50ms 節流

        # 按鈕區域定義（座標區域）
        self.buttons = {
            'play': {'rect': (10, 10, 100, 50), 'label': '[Play]'},
            'pause': {'rect': (120, 10, 210, 50), 'label': '[Pause]'},
            'forward_6': {'rect': (230, 10, 310, 50), 'label': '[+6]'},
            'backward_6': {'rect': (320, 10, 410, 50), 'label': '[-6]'},
            'forward_30': {'rect': (420, 10, 510, 50), 'label': '[+30]'},
            'backward_30': {'rect': (520, 10, 620, 50), 'label': '[-30]'},
            'forward_300': {'rect': (630, 10, 740, 50), 'label': '[+300]'},
            'backward_300': {'rect': (750, 10, 870, 50), 'label': '[-300]'},
            'mark_start': {'rect': (10, 70, 180, 110), 'label': '[Mark Start]'},
            'mark_end': {'rect': (190, 70, 350, 110), 'label': '[Mark End]'},
            'delete_cluster': {'rect': (360, 70, 520, 110), 'label': '[Delete]'},
            'toggle_clahe': {'rect': (530, 70, 680, 110), 'label': '[CLAHE]'},
        }

        # 建立視窗（兩個：雙畫布 + 控制面板）
        cv2.namedWindow('Dual Canvas', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)

        # 註冊滑鼠回調（兩個回調）
        cv2.setMouseCallback('Dual Canvas', self._mouse_callback_canvas)
        cv2.setMouseCallback('Control Panel', self._mouse_callback_control)

    def _mouse_callback_canvas(self, event, x, y, flags, param):
        """雙畫布滑鼠回調（處理輔助線拖曳）"""
        current_time = time.time()

        # 節流（只對 MOUSEMOVE 事件）
        if event == cv2.EVENT_MOUSEMOVE:
            if current_time - self.last_mouse_update < self.mouse_throttle:
                return
            self.last_mouse_update = current_time
            self.mouse_pos_canvas = (x, y)
            # 輔助線拖曳更新（只關心 y 座標）
            self.guide_line.update_position(y)

        # 開始拖曳
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.guide_line.start_dragging(y)

        # 停止拖曳
        elif event == cv2.EVENT_LBUTTONUP:
            self.guide_line.stop_dragging()

    def _mouse_callback_control(self, event, x, y, flags, param):
        """控制面板滑鼠回調（處理按鈕點擊）"""
        current_time = time.time()

        # 更新滑鼠位置（用於懸停效果）
        if event == cv2.EVENT_MOUSEMOVE:
            if current_time - self.last_mouse_update < self.mouse_throttle:
                return
            self.last_mouse_update = current_time
            self.mouse_pos_control = (x, y)

        # 點擊事件：檢測按鈕區域
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._handle_button_click(x, y)

    def _handle_button_click(self, x, y):
        """處理按鈕點擊（檢測按鈕區域）"""
        for btn_name, btn_info in self.buttons.items():
            x1, y1, x2, y2 = btn_info['rect']
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._on_button_click(btn_name)
                break

    def _draw_control_panel(self):
        """繪製控制面板（直接繪製於影像上）"""
        # 建立空白控制面板影像
        panel = np.zeros((150, 900, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # 深灰色背景

        # 繪製所有按鈕
        for btn_name, btn_info in self.buttons.items():
            x1, y1, x2, y2 = btn_info['rect']
            label = btn_info['label']

            # 檢查滑鼠是否懸停
            mx, my = self.mouse_pos_control
            is_hover = x1 <= mx <= x2 and y1 <= my <= y2

            # 按鈕顏色
            color = (100, 200, 100) if is_hover else (80, 80, 80)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(panel, (x1, y1), (x2, y2), (200, 200, 200), 2)

            # 按鈕文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            cv2.putText(panel, label, (text_x, text_y), font, 0.5, (255, 255, 255), 1)

        return panel

    def _draw_status_text(self, frame, frame_idx, time_sec, canvas_side):
        """在影像上繪製狀態文字"""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Frame ID
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   font, 0.8, (255, 255, 255), 2)

        # Time
        cv2.putText(frame, f"Time: {time_sec:.1f}s", (10, 60),
                   font, 0.8, (255, 255, 255), 2)

        # CLAHE 狀態
        clahe_status = "ON" if self.clahe_enabled else "OFF"
        cv2.putText(frame, f"CLAHE: {clahe_status}", (10, 90),
                   font, 0.6, (0, 255, 0) if self.clahe_enabled else (128, 128, 128), 2)

        return frame

    def run(self):
        """主循環"""
        while True:
            # 讀取左右幀
            left_frame = self.frame_reader.get_current_frame()
            right_frame = self.frame_reader.get_frame_at_offset(self.right_offset)

            # CLAHE 處理（可選）
            if self.clahe_enabled:
                left_frame = preprocess_darkroom_frame(left_frame)
                right_frame = preprocess_darkroom_frame(right_frame)

            # 繪製狀態文字
            left_frame = self._draw_status_text(left_frame, left_idx, 'Left')
            right_frame = self._draw_status_text(right_frame, right_idx, 'Right')

            # 繪製輔助線
            left_frame = self.guide_line.draw_on_frame(left_frame)
            right_frame = self.guide_line.draw_on_frame(right_frame)

            # 並排拼接為雙畫布（3840x1080）
            dual_frame = np.hstack([left_frame, right_frame])

            # 顯示雙畫布
            cv2.imshow('Dual Canvas', dual_frame)

            # 繪製並顯示控制面板
            control_panel = self._draw_control_panel()
            cv2.imshow('Control Panel', control_panel)

            # 處理鍵盤輸入（快捷鍵）
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space: 播放/暫停
                self.playing = not self.playing
            elif key == ord('s'):  # S: 儲存
                self._save_progress()
            elif key == ord('z'):  # Z: 復原
                self._undo_last_action()
            elif key == ord('q'):  # Q: 退出
                break
            elif key == ord('c'):  # C: 切換 CLAHE
                self.clahe_enabled = not self.clahe_enabled

            # 播放邏輯
            if self.playing:
                self.frame_reader.read_next_keyframe()
                time.sleep(0.15)  # 150ms 延遲

        cv2.destroyAllWindows()
```

**介面視覺化：**

```
┌──────────────────────────────────────────────────────┐
│  Control Panel (控制面板)                              │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌─────┐ ┌─────┐ │
│  │Play││Pause││ +6 ││ -6 ││ +30││ -30 ││ +300│ │
│  └────┘ └────┘ └────┘ └────┘ └────┘ └─────┘ └─────┘ │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ ┌──────┐        │
│  │Mark Start││ Mark End ││Delete││CLAHE │        │
│  └──────────┘ └──────────┘ └──────┘ └──────┘        │
└──────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Dual Canvas (雙畫布 - 單一視窗 3840x1080)                   │
│  ┌────────────────────────┐  ┌────────────────────────┐   │
│  │ Left: Frame 1200       │  │ Right: Frame 1260      │   │
│  │ Time: 20.0s            │  │ Time: 21.0s (+60)      │   │
│  │ CLAHE: ON              │  │ CLAHE: ON              │   │
│  │                        │  │                        │   │
│  │       ╔═══════╗        │  │       ╔═══════╗        │   │
│  │━━━━━━━║━電梯門━║━━━━━━━━│━━║━━━━━━━║━電梯門━║━━━━━━━│   │
│  │       ╚═══════╝        │  │       ╚═══════╝ (上移) │   │
│  │                        │  │                        │   │
│  │    [影像顯示]           │  │    [影像顯示]           │   │
│  │                        │  │                        │   │
│  └────────────────────────┘  └────────────────────────┘   │
│            ↑ 輔助線橫跨整個視窗，清楚顯示運動              │
└────────────────────────────────────────────────────────────┘
```

**關鍵優勢：**
1. ✅ **高效能**：OpenCV 原生渲染，無 Tkinter 開銷
2. ✅ **簡化實作**：無需 Tkinter widget，直接繪製
3. ✅ **快速響應**：鍵盤快捷鍵即時反應
4. ✅ **防崩潰**：滑鼠回調節流（50ms），避免過度更新
5. ✅ **直覺操作**：滑鼠點擊區域模擬按鈕

**快捷鍵列表：**
- **Space**：播放/暫停
- **S**：儲存進度
- **Z**：復原上一步
- **C**：切換 CLAHE 增強
- **G**：進入/退出輔助線調整模式（可拖曳移動）
- **H**：顯示/隱藏輔助線
- **Q**：退出程式
- **Enter**：標記 Cluster 起始
- **Ctrl + Enter**：確認 Cluster 結束
- **Delete**：刪除當前 Cluster
- **滑鼠拖曳**：調整模式下拖曳輔助線（點擊輔助線附近 ±20px 並拖曳）

**注意：** 幀導航功能改為使用控制面板按鈕（[+6] [-6] [+30] [-30] [+300] [-300]）

**滑鼠節流機制：**
```python
def _mouse_callback_throttled(self, event, x, y, flags, param):
    """
    滑鼠回調節流機制

    目的：
    - 避免滑鼠移動事件過於頻繁導致系統負載過高
    - 防止 GUI 無響應或崩潰

    策略：
    - 設定最小更新間隔（預設 50ms）
    - 只在間隔超過閾值時更新滑鼠位置
    - 點擊事件不受節流影響（即時響應）
    """
    current_time = time.time()

    # 節流：只在間隔超過 50ms 時更新
    if event == cv2.EVENT_MOUSEMOVE:
        if current_time - self.last_mouse_update < 0.05:
            return  # 跳過此次更新

    self.last_mouse_update = current_time
    self.mouse_pos = (x, y)

    # 點擊事件立即處理
    if event == cv2.EVENT_LBUTTONDOWN:
        self._handle_click(x, y)
```

**播放邏輯：**
- 預設速度：每幀停留 150ms（0.6x 速度）
- 快速模式：每幀停留 50ms（2x 速度）
- 右畫布預設跟隨：left_idx + 60（可自訂）
- Cluster 標記中：左畫布固定，右畫布獨立導航
- CLAHE 增強：可開關，預設開啟

---

### 4.3 輔助線系統

**設計目標：提供可調整的水平參考線，協助使用者快速識別運動**

根據參考圖片，黃色水平線可以清楚顯示物體在左右兩幀之間的垂直位移，讓運動事件更容易被識別。

**功能需求：**
1. ✅ 兩種模式與快捷鍵：
   - **G** (Guide)：進入/退出調整模式（可拖曳移動輔助線）
   - **H** (Hide/Show)：顯示/隱藏輔助線
2. ✅ 輔助線橫跨左右兩個畫布（同一 Y 座標）
3. ✅ 調整模式：滑鼠點擊並拖曳調整位置（不需精確）
4. ✅ 輔助線顏色：黃色（高可見度）
5. ✅ 輔助線樣式：實線，寬度 2-3 像素
6. ✅ 播放時輔助線保持固定位置
7. ✅ 輔助線位置記憶（同一影片內保持）

**操作模式：**
- **正常模式**：輔助線固定，可播放影片
- **調整模式（G 鍵）**：進入調整模式，可拖曳輔助線，播放暫停
- **隱藏模式（H 鍵）**：輔助線隱藏，不影響調整模式狀態

**技術實作：**

```python
class GuideLineSystem:
    """
    輔助線系統

    功能：
    - 可調整位置的水平參考線
    - 橫跨左右畫布
    - 協助識別垂直運動

    狀態：
    - visible: 是否顯示輔助線
    - adjustment_mode: 是否進入調整模式（可拖曳）
    """

    def __init__(self, frame_height):
        self.visible = True  # 預設顯示
        self.adjustment_mode = False  # 調整模式（預設關閉）
        self.y_position = frame_height // 2  # 預設位置：畫面中央
        self.frame_height = frame_height
        self.color = (0, 255, 255)  # 黃色 (BGR)
        self.thickness = 2
        self.dragging = False  # 是否正在拖曳

    def toggle_visibility(self):
        """切換輔助線顯示/隱藏（H 鍵）"""
        self.visible = not self.visible
        status = "顯示" if self.visible else "隱藏"
        print(f"🎯 輔助線: {status}")

    def toggle_adjustment_mode(self):
        """切換調整模式（G 鍵）"""
        self.adjustment_mode = not self.adjustment_mode
        status = "調整模式 ON" if self.adjustment_mode else "調整模式 OFF"
        print(f"🎯 輔助線: {status}")

        # 進入調整模式時自動顯示輔助線
        if self.adjustment_mode:
            self.visible = True

    def is_near_line(self, mouse_y, threshold=20):
        """
        檢查滑鼠是否接近輔助線

        Args:
            mouse_y: 滑鼠 Y 座標
            threshold: 檢測範圍（像素）

        Returns:
            bool: 是否在範圍內
        """
        return abs(mouse_y - self.y_position) <= threshold

    def start_dragging(self, mouse_y):
        """
        開始拖曳

        Args:
            mouse_y: 滑鼠 Y 座標
        """
        if self.adjustment_mode and self.is_near_line(mouse_y):
            self.dragging = True
            print(f"🎯 開始拖曳輔助線")

    def update_position(self, mouse_y):
        """
        更新輔助線位置（拖曳中）

        Args:
            mouse_y: 滑鼠 Y 座標
        """
        if self.dragging:
            self.y_position = max(0, min(self.frame_height - 1, mouse_y))

    def stop_dragging(self):
        """停止拖曳"""
        if self.dragging:
            print(f"🎯 輔助線位置: Y={self.y_position}")
            self.dragging = False

    def draw_on_frame(self, frame):
        """
        在幀上繪製輔助線

        Args:
            frame: 輸入影像（BGR）

        Returns:
            frame: 繪製輔助線後的影像
        """
        if not self.visible:
            return frame

        frame_with_line = frame.copy()
        h, w = frame_with_line.shape[:2]

        # 調整模式：使用更亮的顏色和虛線樣式
        if self.adjustment_mode:
            color = (0, 255, 255)  # 亮黃色
            thickness = 3
            # 繪製虛線效果（每 20 像素一段）
            for x in range(0, w, 20):
                cv2.line(frame_with_line,
                        (x, self.y_position),
                        (min(x + 10, w - 1), self.y_position),
                        color, thickness)
        else:
            # 正常模式：實線
            color = self.color
            thickness = self.thickness
            cv2.line(frame_with_line,
                    (0, self.y_position),
                    (w - 1, self.y_position),
                    color, thickness)

        # 在線的兩端繪製小標記（便於識別）
        marker_size = 10
        cv2.line(frame_with_line,
                (0, self.y_position - marker_size),
                (0, self.y_position + marker_size),
                color, thickness + 1)
        cv2.line(frame_with_line,
                (w - 1, self.y_position - marker_size),
                (w - 1, self.y_position + marker_size),
                color, thickness + 1)

        # 調整模式：顯示提示文字
        if self.adjustment_mode:
            cv2.putText(frame_with_line,
                       "Guide Line Adjustment Mode - Drag to adjust",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)

        return frame_with_line

    def get_position(self):
        """取得當前輔助線位置"""
        return self.y_position

    def set_position(self, y):
        """
        設定輔助線位置

        Args:
            y: Y 座標（像素）
        """
        self.y_position = max(0, min(self.frame_height - 1, y))
```

**整合至 OpenCVGUIPlayer：**

```python
class OpenCVGUIPlayer:
    def __init__(self, video_path, scale_factor):
        # ... 原有初始化 ...

        # 初始化輔助線系統
        frame_height = self.frame_height  # 1080
        self.guide_line = GuideLineSystem(frame_height)

        # 註冊滑鼠回調（兩個回調：雙畫布、控制面板）
        cv2.setMouseCallback('Dual Canvas', self._mouse_callback_canvas)
        cv2.setMouseCallback('Control Panel', self._mouse_callback_control)

    def _mouse_callback_canvas(self, event, x, y, flags, param):
        """雙畫布滑鼠回調（處理輔助線拖曳）"""
        current_time = time.time()

        # 節流（滑鼠移動）
        if event == cv2.EVENT_MOUSEMOVE:
            if current_time - self.last_mouse_update < self.mouse_throttle:
                return
            self.last_mouse_update = current_time
            self.mouse_pos_canvas = (x, y)
            # 拖曳中更新輔助線位置（只關心 y 座標）
            self.guide_line.update_position(y)

        # 左鍵按下：開始拖曳輔助線
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.guide_line.start_dragging(y)

        # 左鍵釋放：停止拖曳
        elif event == cv2.EVENT_LBUTTONUP:
            self.guide_line.stop_dragging()

    def run(self):
        """主循環"""
        while True:
            # 讀取左右幀
            left_frame = self.frame_reader.get_current_frame()
            right_frame = self.frame_reader.get_frame_at_offset(self.right_offset)

            # CLAHE 處理（可選）
            if self.clahe_enabled:
                left_frame = preprocess_darkroom_frame(left_frame)
                right_frame = preprocess_darkroom_frame(right_frame)

            # 繪製狀態文字
            left_frame = self._draw_status_text(left_frame, left_idx, 'Left')
            right_frame = self._draw_status_text(right_frame, right_idx, 'Right')

            # 繪製輔助線（如果顯示）
            left_frame = self.guide_line.draw_on_frame(left_frame)
            right_frame = self.guide_line.draw_on_frame(right_frame)

            # 並排拼接為雙畫布（3840x1080）
            dual_frame = np.hstack([left_frame, right_frame])

            # 顯示雙畫布
            cv2.imshow('Dual Canvas', dual_frame)

            # 繪製並顯示控制面板
            control_panel = self._draw_control_panel()
            cv2.imshow('Control Panel', control_panel)

            # 處理鍵盤輸入（快捷鍵）
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space: 播放/暫停
                self.playing = not self.playing
            elif key == ord('g'):  # G: 切換調整模式
                self.guide_line.toggle_adjustment_mode()
                # 進入調整模式時暫停播放
                if self.guide_line.adjustment_mode:
                    self.playing = False
            elif key == ord('h'):  # H: 顯示/隱藏輔助線
                self.guide_line.toggle_visibility()
            elif key == ord('s'):  # S: 儲存
                self._save_progress()
            elif key == ord('z'):  # Z: 復原
                self._undo_last_action()
            elif key == ord('q'):  # Q: 退出
                break
            elif key == ord('c'):  # C: 切換 CLAHE
                self.clahe_enabled = not self.clahe_enabled

            # 播放邏輯（調整模式下不播放）
            if self.playing and not self.guide_line.adjustment_mode:
                self.frame_reader.read_next_keyframe()
                time.sleep(0.15)  # 150ms 延遲

        cv2.destroyAllWindows()
```

**使用者工作流：**

```
1. 使用者正在播放影片
   → 左畫布：Frame 1200
   → 右畫布：Frame 1260
   → 輔助線預設顯示在畫面中央

2. 發現疑似運動區域，按下 [Space] 暫停

3. 按下 [G] 進入調整模式
   → 系統提示：「Guide Line Adjustment Mode - Drag to adjust」
   → 輔助線變為虛線樣式（更亮的黃色）
   → 播放自動暫停

4. 滑鼠拖曳調整輔助線位置：
   方式 A：點擊輔助線附近（±20px 範圍）並拖曳
   → 輔助線即時跟隨滑鼠 Y 座標移動
   → 左右畫布同步調整

   方式 B：直接點擊目標位置附近並拖曳
   → 不需要很精確，範圍內即可開始拖曳

5. 將輔助線對齊至參考物體（例如：電梯門邊緣）
   → 左畫布：輔助線對齊物體邊緣
   → 右畫布：輔助線位置相同，可看出物體位移
   → 釋放滑鼠左鍵完成調整

6. 按下 [G] 退出調整模式
   → 輔助線恢復為實線樣式
   → 提示文字消失
   → 輔助線位置固定

7. 按下 [Space] 繼續播放
   → 輔助線保持固定位置
   → 使用者可輕鬆觀察物體相對於輔助線的移動

8. 如果需要再次調整：
   → 按 [Space] 暫停
   → 按 [G] 進入調整模式
   → 拖曳輔助線至新位置
   → 按 [G] 退出調整模式

9. 如不想看到輔助線（但保留位置）：
   → 按 [H] 隱藏輔助線
   → 輔助線消失，但位置記憶保留
   → 再按 [H] 重新顯示於原位置

10. 調整模式與顯示獨立：
    → 可以在隱藏狀態下進入調整模式（[G]）
    → 進入調整模式會自動顯示輔助線
    → 退出調整模式不影響顯示/隱藏狀態
```

**視覺化範例：**

```
啟用輔助線前：
┌─────────────────────────┐  ┌─────────────────────────┐
│  Left Canvas            │  │  Right Canvas           │
│  Frame: 1200            │  │  Frame: 1260            │
│                         │  │                         │
│     ╔═══════╗           │  │     ╔═══════╗           │
│     ║ 電梯門 ║           │  │     ║ 電梯門 ║           │
│     ╚═══════╝           │  │     ╚═══════╝           │
│                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘

啟用輔助線後（按 H）：
┌─────────────────────────┐  ┌─────────────────────────┐
│  Left Canvas            │  │  Right Canvas           │
│  Frame: 1200            │  │  Frame: 1260            │
│                         │  │                         │
│     ╔═══════╗           │  │     ╔═══════╗           │
│━━━━━║━電梯門━║━━━━━━━━━━│  │━━━━━║━電梯門━║━━━━━━━━━━│ ← 黃色輔助線
│     ╚═══════╝           │  │     ╚═══════╝           │
│                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘

調整輔助線位置（按 ↑ 多次）：
┌─────────────────────────┐  ┌─────────────────────────┐
│  Left Canvas            │  │  Right Canvas           │
│  Frame: 1200            │  │  Frame: 1260            │
│━━━━━╔═══════╗━━━━━━━━━━│  │━━━━━╔═══════╗━━━━━━━━━━│ ← 對齊門上緣
│     ║ 電梯門 ║           │  │     ║ 電梯門 ║ (上移)   │
│     ╚═══════╝           │  │     ╚═══════╝           │
│                         │  │                         │
└─────────────────────────┘  └─────────────────────────┘
                                  ↑ 可看出右邊門上緣相對上移
```

**進階功能（可選）：**

1. **多條輔助線**
   - 支援同時顯示 2-3 條輔助線
   - 快捷鍵：H（主線）、H+1（輔助線1）、H+2（輔助線2）
   - 不同顏色區分：黃色、青色、品紅

2. **輔助線標籤**
   - 在輔助線旁顯示 Y 座標
   - 例如：`━━━━━━ Y=325 ━━━━━━`

3. **滑鼠拖曳調整**
   - 在畫布上點擊並拖曳輔助線
   - 更直覺的位置調整方式

4. **輔助線位置儲存**
   - 將輔助線位置儲存至 JSON 檔案
   - 下次開啟同一影片時自動載入

**快捷鍵更新（含輔助線）：**
- **G** (Guide)：進入/退出輔助線調整模式
  - 進入調整模式時自動暫停播放
  - 可拖曳移動輔助線
  - 輔助線變為虛線樣式
- **H** (Hide/Show)：顯示/隱藏輔助線
  - 獨立於調整模式
  - 隱藏後位置保留
- **滑鼠拖曳**：調整模式下拖曳輔助線
  - 點擊輔助線附近（±20px）並拖曳
  - 即時更新位置
  - 左右畫布同步

**注意事項：**
- 調整模式下自動暫停播放，退出後可繼續播放
- 進入調整模式時會自動顯示輔助線（即使之前隱藏）
- 幀導航使用控制面板按鈕（[+6] [-6] [+30] [-30] [+300] [-300]）

---

### 4.4 Cluster 標記工作流

**工作流程：**

```
1. [瀏覽模式] 使用者前後導航，尋找運動事件
   ↓
2. [標記起始] 按下「標記 Cluster 起始」
   - 生成 cluster_id = physical_cluster_counter + 1
   - 記錄 cluster_start_idx (左畫布當前幀)
   - 匯出 pre_cluster_XXX.jpg (左畫布)
   - 鎖定左畫布
   ↓
3. [尋找結束] 右畫布獨立導航，尋找運動結束幀
   - 支援所有導航控制（±6/±30/±300）
   - 使用順序讀取，不使用 OpenCV 跳轉
   - 左畫布保持固定顯示
   ↓
4. [標記結束] 按下「確認 Cluster 結束」
   - 記錄 cluster_end_idx (右畫布當前幀)
   - 匯出 post_cluster_XXX.jpg (右畫布)
   ↓
5. [線段標記] 自動進入線段標記模式
   - 步驟 5.1：選擇 ROI 區域（在左畫布拖曳，同步顯示在右畫布）
   - 步驟 5.2：3x 放大顯示 ROI（左右並排）
   - 步驟 5.3：標記 3 次，每次分別在左右畫布標記線段
   - 步驟 5.4：計算線段 Y 分量變化（Δy = y_right - y_left）
   - 步驟 5.5：取平均並計算標準差
   ↓
6. [自動判定方向] 根據線段 Y 分量變化計算方向
   - orientation = sign(Δy)
   - 如果 Δy > 0（Y 分量增加）→ DOWN（向下移動）
   - 如果 Δy < 0（Y 分量減少）→ UP（向上移動）
   ↓
7. [確認對話框] 顯示計算結果供使用者確認
   - 提示：「cluster_XX: Δy=+/-YY.YY px, ZZ.ZZ mm UP/DOWN, average AA.AAA mm / 6 frames」
   - 使用者可選擇：[確認] [取消並重新標記]
   ↓
8. [等速分配] 計算每個關鍵幀的位移
   - num_keyframes = (end_idx - start_idx) / 6
   - displacement_per_frame = total_mm / num_keyframes
   ↓
9. [寫入 CSV] 即時寫入結果
   - 支援中斷繼續
   ↓
10. [繼續標記] 解鎖畫布，回到瀏覽模式
```

**錯誤處理機制：**
- **清除當前線段重繪**：在三次標記過程中，可清除當前標記並重新繪製
- **刪除整個 Cluster**：標記完成後發現錯誤，可刪除整個 Cluster（包含 CSV 記錄和 JPG 檔案）

---

### 4.4 線段標記法

**參考 `manual_correction_tool.py` 的實作：**

```python
class LineSegmentMarker:
    def __init__(self, left_frame, right_frame):
        self.left_frame = left_frame
        self.right_frame = right_frame
        self.measurements = []  # 儲存 3 次測量
        self.zoom_factor = 3  # 3x 放大
        self.roi_rect = None  # 共用的 ROI 區域 (x, y, w, h)

    def select_roi_on_left(self):
        """
        在左畫布上選擇 ROI，同步顯示在左右畫布

        工作流程：
        1. 在左畫布上拖曳選擇 ROI 矩形
        2. 同步在右畫布上顯示相同位置的 ROI 矩形框（紅色虛線）
        3. 使用者確認後，左右畫布同時更新為放大後的 ROI（並排顯示）

        Returns:
            (roi_x, roi_y, roi_w, roi_h): ROI 矩形座標
        """
        # 建立並排顯示（用於選擇 ROI）
        dual_canvas = np.hstack([self.left_frame, self.right_frame])

        # 使用者在左畫布拖曳選擇 ROI
        roi_x, roi_y, roi_w, roi_h = self._interactive_roi_selection(dual_canvas)

        # 約束 ROI 尺寸（確保放大後不超出螢幕）
        MAX_ROI_SIZE = 600  # 像素（3x 放大後 = 1800，並排 3600 < 3840 ✅）
        MIN_ROI_SIZE = 100  # 像素

        if roi_w < MIN_ROI_SIZE or roi_h < MIN_ROI_SIZE:
            raise ValueError(f"ROI 區域太小（{roi_w}x{roi_h}），請重新選擇")

        if roi_w > MAX_ROI_SIZE or roi_h > MAX_ROI_SIZE:
            scale = min(MAX_ROI_SIZE / roi_w, MAX_ROI_SIZE / roi_h)
            roi_w = int(roi_w * scale)
            roi_h = int(roi_h * scale)
            print(f"⚠️ ROI 已調整為 {roi_w}x{roi_h} 以確保放大效果")

        self.roi_rect = (roi_x, roi_y, roi_w, roi_h)
        return self.roi_rect

    def show_zoomed_roi_dual_canvas(self):
        """
        顯示放大後的 ROI（左右並排）

        Returns:
            dual_zoomed_canvas: 並排的放大 ROI（例如 1800x900）
        """
        roi_x, roi_y, roi_w, roi_h = self.roi_rect

        # 提取 ROI
        left_roi = self.left_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        right_roi = self.right_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # 3x 放大
        left_zoomed = cv2.resize(left_roi, None, fx=self.zoom_factor, fy=self.zoom_factor,
                                 interpolation=cv2.INTER_LINEAR)
        right_zoomed = cv2.resize(right_roi, None, fx=self.zoom_factor, fy=self.zoom_factor,
                                  interpolation=cv2.INTER_LINEAR)

        # 並排顯示
        dual_zoomed_canvas = np.hstack([left_zoomed, right_zoomed])

        return dual_zoomed_canvas

    def mark_line_segment_on_zoomed_canvas(self, canvas_side):
        """
        在放大後的畫布上標記線段

        Args:
            canvas_side: 'left' or 'right'

        Returns:
            (point1, point2): 線段兩端點（原始影像座標）
        """
        # 顯示放大後的並排畫布
        dual_zoomed = self.show_zoomed_roi_dual_canvas()

        # 等待使用者點選兩個點
        print(f"請在 {canvas_side} 畫布上標記線段起點")
        point1_zoomed = self._wait_for_click_on_dual_canvas(dual_zoomed, canvas_side)

        # 繪製第一個點
        cv2.circle(dual_zoomed, point1_zoomed, 5, (0, 255, 0), -1)
        cv2.imshow('Zoomed ROI - Line Marking', dual_zoomed)

        print(f"請在 {canvas_side} 畫布上標記線段終點")
        point2_zoomed = self._wait_for_click_on_dual_canvas(dual_zoomed, canvas_side)

        # 繪製線段
        cv2.line(dual_zoomed, point1_zoomed, point2_zoomed, (0, 255, 0), 2)
        cv2.imshow('Zoomed ROI - Line Marking', dual_zoomed)

        # 確認標記
        if not self._confirm_marking():
            return self.mark_line_segment_on_zoomed_canvas(canvas_side)

        # 轉換回原始座標
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        original_point1 = (
            roi_x + point1_zoomed[0] / self.zoom_factor,
            roi_y + point1_zoomed[1] / self.zoom_factor
        )
        original_point2 = (
            roi_x + point2_zoomed[0] / self.zoom_factor,
            roi_y + point2_zoomed[1] / self.zoom_factor
        )

        return original_point1, original_point2

    def perform_three_measurements(self):
        """
        標記 3 次，取平均

        測量方式：計算線段 Y 分量的差異
        - 左畫布線段 Y 分量：y_left
        - 右畫布線段 Y 分量：y_right
        - Y 分量變化：Δy = y_right - y_left
        """
        self.measurements = []

        # 第一次：選擇 ROI（左右畫布共用）
        print("=" * 50)
        print("步驟 1：選擇 ROI 區域")
        print("=" * 50)
        self.select_roi_on_left()

        for i in range(3):
            print(f"\n第 {i+1}/3 次測量")

            # 在左畫布標記線段
            left_line = self.mark_line_segment_on_zoomed_canvas('left')

            # 在右畫布標記線段
            right_line = self.mark_line_segment_on_zoomed_canvas('right')

            # 計算線段 Y 分量（取絕對值）
            left_y_component = abs(left_line[1][1] - left_line[0][1])
            right_y_component = abs(right_line[1][1] - right_line[0][1])

            # 計算 Y 分量變化
            d_y_component = right_y_component - left_y_component

            self.measurements.append(d_y_component)
            print(f"  左畫布線段 Y 分量: {left_y_component:.2f} 像素")
            print(f"  右畫布線段 Y 分量: {right_y_component:.2f} 像素")
            print(f"  Y 分量變化: {d_y_component:.2f} 像素")

        # 計算平均與標準差
        avg_d_y_px = np.mean(self.measurements)
        std_d_y_px = np.std(self.measurements)

        print(f"\n測量結果：平均 Y 分量變化 {avg_d_y_px:.2f} ± {std_d_y_px:.2f} 像素")

        # 警告標準差過大
        if std_d_y_px > 2.0:
            print(f"⚠️ 警告：標準差較大 ({std_d_y_px:.2f} px)，建議重新測量")

        return avg_d_y_px

    def _confirm_marking(self):
        """
        顯示確認對話框

        Returns:
            True: 確認
            False: 清除重繪
        """
        # 使用 Tkinter messagebox
        result = messagebox.askyesno(
            "確認標記",
            "線段標記是否正確？\n\n點選「是」確認，點選「否」清除重繪"
        )
        return result
```

**標記流程視覺化（更新為 3x 放大與並排顯示）：**

```
步驟 1：ROI 選擇（在左畫布拖曳，同步顯示在右畫布）
┌──────────────────────────────────────────────────────────┐
│  並排顯示（原始尺寸 1920x1080 + 1920x1080）                │
│  ┌────────────────────────┐  ┌────────────────────────┐   │
│  │ 左畫布（起始幀）        │  │ 右畫布（結束幀）        │   │
│  │                        │  │                        │   │
│  │     ╔═══════╗          │  │     ╔═══════╗          │   │
│  │     ║ 電梯門 ║          │  │     ║ 電梯門 ║          │   │
│  │     ╠───────╣ ← 拖曳   │  │     ╠───────╣ ← 同步   │   │
│  │     ║ 選擇  ║   ROI    │  │     ║ 顯示  ║   ROI    │   │
│  │     ╚═══════╝          │  │     ╚═══════╝          │   │
│  │       [ROI]            │  │       [ROI]            │   │
│  └────────────────────────┘  └────────────────────────┘   │
└──────────────────────────────────────────────────────────┘

步驟 2：3x 放大後並排顯示（例如 300x300 ROI → 900x900，並排 1800x900）
┌──────────────────────────────────────────────────────────┐
│  放大 3x 並排顯示（1800x900，遠小於 3840 寬度 ✅）         │
│  ┌────────────────────────┐  ┌────────────────────────┐   │
│  │ 左 ROI (3x)            │  │ 右 ROI (3x)            │   │
│  │                        │  │                        │   │
│  │   ╔═══════════════╗    │  │   ╔═══════════════╗    │   │
│  │   ║               ║    │  │   ║               ║    │   │
│  │   ║  電梯門邊緣    ║    │  │   ║  電梯門邊緣    ║    │   │
│  │   ║               ║    │  │   ║               ║    │   │
│  │   ╠●══════════════╣ ←  │  │   ╠●══════════════╣ ←  │   │
│  │   ║ │標記線段     ║    │  │   ║ │標記線段     ║    │   │
│  │   ║●══════════════╣    │  │   ║●══════════════╣    │   │
│  │   ╚═══════════════╝    │  │   ╚═══════════════╝    │   │
│  │                        │  │                        │   │
│  └────────────────────────┘  └────────────────────────┘   │
│        y_left = 100px             y_right = 130px         │
│                     Δy = +30px (增加)                     │
└──────────────────────────────────────────────────────────┘

   [確認] [清除重繪]
```

---

### 4.5 自動方向判定與確認對話框

```python
def auto_determine_orientation_and_confirm(cluster_id, avg_d_y_px, scale_factor, num_keyframes):
    """
    自動判定方向並顯示確認對話框

    測量方式：基於線段 Y 分量變化
    - 左畫布線段 Y 分量：y_left（起始幀，參考）
    - 右畫布線段 Y 分量：y_right（結束幀，運動後）
    - Y 分量變化：Δy = y_right - y_left

    Args:
        cluster_id: Cluster 編號
        avg_d_y_px: 平均線段 Y 分量變化（像素，可正可負）
        scale_factor: 比例尺（像素/10mm）
        num_keyframes: 關鍵幀數量

    Returns:
        (confirmed, orientation, total_mm, avg_mm_per_frame)
    """
    # 計算總位移（mm）
    total_mm = abs(avg_d_y_px) * 10 / scale_factor

    # 判定方向（基於線段 Y 分量變化）
    #
    # 物理原理：
    # - 如果 Δy > 0（Y 分量增加），表示向下移動 → DOWN
    # - 如果 Δy < 0（Y 分量減少），表示向上移動 → UP
    #
    # 注意：實際方向判定取決於：
    # 1. 相機的安裝位置和方向
    # 2. 標記的參考構造（門框 vs 其他固定結構）
    # 3. Y 軸座標系統的定義
    #
    # 建議：首次使用時，根據已知運動方向校準此判定邏輯
    if avg_d_y_px > 0:
        orientation = -1  # DOWN（Y 分量增加）
        direction_text = "DOWN"
    else:
        orientation = 1   # UP（Y 分量減少）
        direction_text = "UP"

    # 計算每幀平均位移
    avg_mm_per_frame = total_mm / num_keyframes if num_keyframes > 0 else 0

    # 顯示確認對話框
    message = (
        f"計算結果：\n\n"
        f"Cluster #{cluster_id:03d}\n"
        f"Y 分量變化: {avg_d_y_px:+.2f} 像素\n"
        f"總位移: {total_mm:.2f} mm {direction_text}\n"
        f"關鍵幀數: {num_keyframes} 幀\n"
        f"平均: {avg_mm_per_frame:.3f} mm / 6 frames\n\n"
        f"是否確認此標記結果？"
    )

    confirmed = messagebox.askyesno("確認 Cluster 標記", message)

    if confirmed:
        print(f"✅ Cluster #{cluster_id:03d}: Δy={avg_d_y_px:+.2f}px, "
              f"{total_mm:.2f} mm {direction_text}, "
              f"average {avg_mm_per_frame:.3f} mm / 6 frames")
    else:
        print(f"❌ 使用者取消 Cluster #{cluster_id:03d} 標記")

    return confirmed, orientation, total_mm, avg_mm_per_frame
```

---

### 4.6 等速分配演算法

**設計理念：**
- 假設運動在起始幀與結束幀之間為等速運動
- 將總位移平均分配給每個關鍵幀（6的倍數）

**實作：**

```python
def distribute_displacement_uniformly(start_idx, end_idx, total_mm, orientation, frame_interval=6):
    """
    等速分配位移至關鍵幀

    重要：start_idx 和 end_idx 是參考幀（靜止），位移分配到兩者之間的關鍵幀

    Args:
        start_idx: 運動起始參考幀索引（靜止）
        end_idx: 運動結束參考幀索引（靜止）
        total_mm: 總位移（mm，絕對值）
        orientation: 方向（1=UP, -1=DOWN）
        frame_interval: 關鍵幀間隔

    Returns:
        dict: {frame_idx: displacement_mm} (帶符號)
    """
    # 計算關鍵幀範圍
    # 起始幀的下一個關鍵幀 = ((start_idx // 6) + 1) * 6
    # 結束幀的前一個關鍵幀 = ((end_idx // 6) - 1) * 6
    start_keyframe = ((start_idx // frame_interval) + 1) * frame_interval
    end_keyframe = ((end_idx // frame_interval) - 1) * frame_interval

    # 如果 end_keyframe < start_keyframe，表示兩個參考幀之間沒有關鍵幀
    if end_keyframe < start_keyframe:
        print(f"⚠️ 警告：參考幀 {start_idx} 和 {end_idx} 之間沒有關鍵幀，無法分配位移")
        return {}

    keyframes = list(range(start_keyframe, end_keyframe + 1, frame_interval))
    num_keyframes = len(keyframes)

    if num_keyframes == 0:
        return {}

    # 平均分配（帶符號）
    displacement_per_frame = (total_mm / num_keyframes) * orientation

    result = {}
    for frame_idx in keyframes:
        result[frame_idx] = displacement_per_frame

    print(f"📊 等速分配：{num_keyframes} 幀，每幀 {displacement_per_frame:.3f} mm")

    return result

# 範例
# start_idx = 1200 (參考幀，靜止), end_idx = 1260 (參考幀，靜止), total_mm = 30.0, orientation = 1 (UP)
# start_keyframe = 1206, end_keyframe = 1254
# keyframes = [1206, 1212, 1218, 1224, 1230, 1236, 1242, 1248, 1254]
# num_keyframes = 9
# displacement_per_frame = (30.0 / 9) * 1 = +3.333 mm (向上)
```

---

### 4.7 增量 CSV 寫入

**CSV 結構（調整後）：**

```csv
frame_idx,second,vertical_travel_distance_mm,cluster_id,orientation,frame_path,marking_status
1200,20.000,0.0,0,0,pre_cluster_001.jpg,manual
1206,20.100,3.333,1,1,,manual
1212,20.200,3.333,1,1,,manual
1218,20.300,3.333,1,1,,manual
1224,20.400,3.333,1,1,,manual
1230,20.500,3.333,1,1,,manual
1236,20.600,3.333,1,1,,manual
1242,20.700,3.333,1,1,,manual
1248,20.800,3.333,1,1,,manual
1254,20.900,3.333,1,1,,manual
1260,21.000,0.0,0,0,post_cluster_001.jpg,manual
1266,21.100,0.0,0,0,,auto
```

**欄位說明：**
- `frame_idx`: 幀索引
- `second`: 時間戳（秒）
- `vertical_travel_distance_mm`: 垂直位移（mm，帶符號）
- `cluster_id`: 群集編號（0 = 群外）
- `orientation`: 方向（1=UP, -1=DOWN, 0=群外）
- `frame_path`: JPG 檔名（pre/post）
- `marking_status`: `manual`（人工標記）或 `auto`（自動填充）

**增量寫入策略：**

```python
class IncrementalCSVWriter:
    def __init__(self, csv_path, fps, frame_interval=6):
        self.csv_path = csv_path
        self.fps = fps
        self.frame_interval = frame_interval
        self.data = self._load_existing_or_init()

    def _load_existing_or_init(self):
        """載入已存在的 CSV 或初始化"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            print(f"📂 載入既有進度：{len(df)} 筆記錄")
            return df
        else:
            return pd.DataFrame(columns=[
                'frame_idx', 'second', 'vertical_travel_distance_mm',
                'cluster_id', 'orientation', 'frame_path', 'marking_status'
            ])

    def get_last_processed_frame(self):
        """取得上次處理的最後一幀"""
        if len(self.data) == 0:
            return 0
        return int(self.data['frame_idx'].max())

    def get_max_cluster_id(self):
        """取得最大的 cluster_id"""
        if len(self.data) == 0:
            return 0
        return int(self.data['cluster_id'].max())

    def append_cluster(self, cluster_data):
        """
        新增 cluster 資料

        Args:
            cluster_data: {
                'cluster_id': 1,
                'start_idx': 1200,  ← 參考幀（靜止）
                'end_idx': 1260,    ← 參考幀（靜止）
                'displacement_dict': {1206: 3.333, 1212: 3.333, ..., 1254: 3.333},
                'orientation': 1,
                'pre_jpg': 'pre_cluster_001.jpg',
                'post_jpg': 'post_cluster_001.jpg'
            }
        """
        new_rows = []

        # 先寫入 start_idx 參考幀（群外，帶 pre JPG）
        new_rows.append({
            'frame_idx': cluster_data['start_idx'],
            'second': round(cluster_data['start_idx'] / self.fps, 3),
            'vertical_travel_distance_mm': 0.0,  # 靜止
            'cluster_id': 0,  # 群外
            'orientation': 0,  # 群外
            'frame_path': cluster_data['pre_jpg'],
            'marking_status': 'manual'
        })

        # 寫入所有運動幀
        for frame_idx, displacement in cluster_data['displacement_dict'].items():
            new_rows.append({
                'frame_idx': frame_idx,
                'second': round(frame_idx / self.fps, 3),
                'vertical_travel_distance_mm': round(displacement, 3),
                'cluster_id': cluster_data['cluster_id'],
                'orientation': cluster_data['orientation'],
                'frame_path': '',  # 運動幀不標記 JPG
                'marking_status': 'manual'
            })

        # 最後寫入 end_idx 參考幀（群外，帶 post JPG）
        new_rows.append({
            'frame_idx': cluster_data['end_idx'],
            'second': round(cluster_data['end_idx'] / self.fps, 3),
            'vertical_travel_distance_mm': 0.0,  # 靜止
            'cluster_id': 0,  # 群外
            'orientation': 0,  # 群外
            'frame_path': cluster_data['post_jpg'],
            'marking_status': 'manual'
        })

        self.data = pd.concat([self.data, pd.DataFrame(new_rows)], ignore_index=True)
        self.data.sort_values('frame_idx', inplace=True)
        self.data.drop_duplicates(subset='frame_idx', keep='last', inplace=True)
        self.save()

    def delete_cluster(self, cluster_id):
        """
        刪除指定的 cluster

        Args:
            cluster_id: 要刪除的 cluster 編號
        """
        self.data = self.data[self.data['cluster_id'] != cluster_id]
        self.save()
        print(f"🗑️ 已刪除 Cluster #{cluster_id:03d}")

    def save(self):
        """即時儲存"""
        self.data.to_csv(self.csv_path, index=False)
        print(f"💾 已儲存進度至 {self.csv_path}")
```

---

### 4.8 Cluster 管理器（協調 CSV 與 JPG 刪除）

**設計目標：確保資料一致性，避免 CSV 與 JPG 不同步**

**實作：**

```python
import os

class ClusterManager:
    """
    Cluster 管理器

    職責：
    - 協調 CSV 與 JPG 的刪除操作
    - 確保資料一致性（先查詢 JPG → 刪除 JPG → 刪除 CSV）
    - 錯誤處理與使用者回饋
    """

    def __init__(self, csv_writer, export_dir):
        """
        Args:
            csv_writer: IncrementalCSVWriter 實例
            export_dir: JPG 匯出目錄（例如：lifts/exported_frames/21a_dark/）
        """
        self.csv_writer = csv_writer
        self.export_dir = export_dir

    def delete_cluster(self, cluster_id):
        """
        刪除 Cluster（包含 CSV 記錄與 JPG 檔案）

        步驟：
        1. 從 CSV 查詢要刪除的 JPG 檔案路徑
        2. 刪除 JPG 檔案（pre + post）
        3. 刪除 CSV 記錄（包含參考幀）
        4. 顯示確認訊息

        Args:
            cluster_id: 要刪除的 cluster 編號
        """
        # 步驟 1：查詢要刪除的 JPG 檔案
        # 注意：參考幀的 cluster_id = 0，但 frame_path 有值
        # 需要查詢 pre_cluster_XXX.jpg 和 post_cluster_XXX.jpg
        cluster_rows = self.csv_writer.data[
            self.csv_writer.data['frame_path'].str.contains(
                f'cluster_{cluster_id:03d}.jpg',
                na=False
            )
        ]

        jpg_files = cluster_rows['frame_path'].tolist()

        # 步驟 2：刪除 JPG 檔案
        deleted_count = 0
        for jpg_file in jpg_files:
            jpg_path = os.path.join(self.export_dir, jpg_file)
            if os.path.exists(jpg_path):
                try:
                    os.remove(jpg_path)
                    print(f"🗑️  已刪除 JPG: {jpg_file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 刪除 JPG 失敗: {jpg_file} ({e})")
            else:
                print(f"⚠️  JPG 不存在: {jpg_file}")

        # 步驟 3：刪除 CSV 記錄
        # IncrementalCSVWriter.delete_cluster() 只刪除 cluster_id 匹配的運動幀
        # 需要額外刪除參考幀（cluster_id=0 但 frame_path 包含此 cluster）
        self.csv_writer.data = self.csv_writer.data[
            ~self.csv_writer.data['frame_path'].str.contains(
                f'cluster_{cluster_id:03d}.jpg',
                na=False
            )
        ]
        self.csv_writer.data = self.csv_writer.data[
            self.csv_writer.data['cluster_id'] != cluster_id
        ]
        self.csv_writer.save()

        # 步驟 4：顯示確認訊息
        print(f"✅ Cluster #{cluster_id:03d} 已完全刪除")
        print(f"   - 刪除 {deleted_count} 個 JPG 檔案")
        print(f"   - 刪除 CSV 記錄")
```

**使用範例：**

```python
class OpenCVGUIPlayer:
    def __init__(self, video_path, scale_factor):
        # ... 原有初始化 ...

        # 建立 CSV writer
        self.csv_writer = IncrementalCSVWriter(csv_path, fps)

        # 建立 Cluster 管理器
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        export_dir = f'lifts/exported_frames/{video_name}_dark'
        self.cluster_manager = ClusterManager(self.csv_writer, export_dir)

    def _on_button_click(self, btn_name):
        """處理按鈕點擊事件"""
        if btn_name == 'delete_cluster':
            if self.current_cluster_id:
                # 顯示確認對話框
                import tkinter.messagebox as messagebox
                confirmed = messagebox.askyesno(
                    "刪除 Cluster",
                    f"確定要刪除 Cluster #{self.current_cluster_id:03d} 嗎？\n\n"
                    f"此操作將刪除：\n"
                    f"- CSV 記錄（運動幀 + 參考幀）\n"
                    f"- JPG 檔案（pre + post）\n\n"
                    f"此操作無法復原。"
                )

                if confirmed:
                    # 使用 ClusterManager 統一處理
                    self.cluster_manager.delete_cluster(self.current_cluster_id)
                    self.current_cluster_id = None
                else:
                    print("❌ 取消刪除操作")
            else:
                print("⚠️  沒有選擇要刪除的 Cluster")
```

---

## 5. 使用者工作流範例

### 5.1 完整標記流程

```
1. 啟動程式
   $ uv run python src/lift_travel_detection_dark.py

2. 選擇影片
   → 程式顯示：請選擇暗房影片檔案
   → 使用者選擇：21a.mp4

3. 載入影片與設定
   → 載入比例尺快取：21.mp4 → 45.2 px/10mm
   → 載入暗房區間：20-40s, 60-80s
   → 檢查既有進度：lifts/result/21a_dark.csv
   → 發現既有進度：已標記 2 個 cluster
   → 順序讀取至上次位置：frame 1500

4. 進入 GUI 播放器
   → 左畫布：Frame 1500 (25.0s)
   → 右畫布：Frame 1560 (26.0s, +60)
   → 狀態：已標記 2 個 cluster
   → CLAHE 增強：已開啟

5. 使用者瀏覽影片
   → 按下 [播放 ▶]，系統以 150ms/幀播放
   → 發現疑似運動事件，按下 [暫停 ⏸]
   → 微調：[◀ -6] [◀ -6] 回退 12 幀（順序讀取）

6. 標記 Cluster 起始
   → 按下 [標記 Cluster 起始]
   → 系統：Cluster #003 已建立
   → 系統：pre_cluster_003.jpg 已匯出
   → 左畫布鎖定於 Frame 1680

7. 尋找 Cluster 結束
   → 右畫布獨立導航（順序讀取）
   → 使用 [▶▶ +30] [▶▶ +30] 快速前進
   → 使用 [◀ -6] 微調
   → 確認 Frame 1800 為結束幀

8. 確認 Cluster 結束
   → 按下 [確認 Cluster 結束]
   → 系統：post_cluster_003.jpg 已匯出
   → 系統：進入線段標記模式

9. 線段標記（第 1/3 次）
   → 系統提示：請在左畫布標記參考線段
   → 使用者點選 ROI → 8x 放大 → 標記線段
   → 確認對話框：[確認] [清除重繪]
   → 使用者確認
   → 系統提示：請在右畫布標記對應線段
   → 使用者點選 ROI → 8x 放大 → 標記對應線段
   → 確認對話框：[確認] [清除重繪]
   → 使用者確認
   → 系統：第 1 次測量完成，位移 = -28.5 px

10. 重複標記 2 次
    → 第 2 次：-28.3 px
    → 第 3 次：-28.7 px
    → 平均：-28.5 ± 0.2 px

11. 自動方向判定
    → 像素轉 mm：|-28.5| × 10 / 45.2 = 6.3 mm
    → 方向判定：dy < 0 → UP (orientation = 1)
    → 關鍵幀：1686, 1692, 1698, ..., 1794 (共 19 幀)
    → 每幀分配：6.3 / 19 = 0.332 mm

12. 確認對話框
    → 顯示：「cluster_003: 6.3 mm UP, average 0.332 mm / 6 frames」
    → 使用者按下 [確認]

13. 寫入 CSV
    → 系統：已儲存 Cluster #003 至 21a_dark.csv
    → 系統：已標記 3 個 cluster

14. 繼續標記
    → 畫布解鎖，回到瀏覽模式
    → 使用者可繼續尋找下一個運動事件

15. 發現標記錯誤
    → 使用者發現 Cluster #003 標記錯誤
    → 按下 [刪除當前 Cluster]
    → 系統：已刪除 Cluster #003 及相關檔案
    → 使用者重新標記

16. 結束標記
    → 按下 [Ctrl+Q] 結束程式
    → 系統：最終結果已儲存至 21a_dark.csv
```

---

## 6. 實作階段規劃

### 階段 1：核心基礎重構（1-2 天）
- [ ] 備份現有程式碼
- [ ] 移除自動偵測相關程式碼
- [ ] 保留比例尺快取系統
- [ ] 保留影片旋轉與前處理
- [ ] 保留 JPG 匯出功能
- [ ] 調整 CSV 輸出結構
- [ ] 停用 inspection 影片生成

### 階段 2：逐幀讀取器實作（2-3 天）
- [ ] 實作 `SequentialFrameReader` 類別
- [ ] 實作幀快取機制
- [ ] 實作 `seek_to_frame()` 順序讀取
- [ ] 實作 `get_frame_at_offset()` 偏移讀取
- [ ] 實作向後導航（重新開啟影片）
- [ ] 測試幀讀取準確性（與實際幀索引比對）

### 階段 3：OpenCV GUI 播放器基礎（3-4 天）
- [ ] OpenCV 視窗建立與配置（三視窗：左畫布、右畫布、控制面板）
- [ ] 整合 `SequentialFrameReader`
- [ ] 滑鼠回調節流機制實作（防崩潰）
- [ ] 按鈕繪製與點擊區域檢測
- [ ] 播放控制邏輯（播放/暫停/反向/變速）
- [ ] 微調導航功能（±6/±30/±300，順序讀取）
- [ ] 快捷鍵系統實作（Space, S, Z, C, H, Q, 方向鍵等）
- [ ] 輔助線系統實作（GuideLineSystem 類別）
- [ ] 可自訂右畫布對照間隔
- [ ] CLAHE 開關功能
- [ ] 狀態文字直接繪製於影像上

### 階段 4：Cluster 標記工作流（2-3 天）
- [ ] 標記起始/結束按鈕邏輯
- [ ] 畫布鎖定/解鎖機制
- [ ] 右畫布獨立導航
- [ ] JPG 匯出整合
- [ ] Cluster ID 管理
- [ ] 狀態機實作（瀏覽/標記起始/標記結束/線段標記）

### 階段 5：線段標記整合（3-4 天）
- [ ] 參考 `manual_correction_tool.py` 改寫
- [ ] ROI 選擇功能
- [ ] 8x 放大顯示
- [ ] 線段標記（點選兩端點）
- [ ] 確認對話框（清除重繪功能）
- [ ] 三次測量取平均
- [ ] 像素轉 mm 計算
- [ ] 標準差警告

### 階段 6：自動方向判定與確認（1-2 天）
- [ ] 自動計算 orientation
- [ ] 確認對話框設計
- [ ] 顯示計算結果（總位移、方向、平均）
- [ ] 取消並重新標記功能

### 階段 7：等速分配與 CSV 寫入（2-3 天）
- [ ] 等速分配演算法實作
- [ ] `IncrementalCSVWriter` 類別
- [ ] 進度載入功能（取得上次位置）
- [ ] 順序讀取至上次位置
- [ ] append_cluster() 實作
- [ ] delete_cluster() 實作
- [ ] 即時儲存機制

### 階段 8：錯誤處理與刪除功能（1-2 天）
- [ ] 刪除 Cluster 按鈕
- [ ] 刪除 CSV 記錄
- [ ] 刪除對應 JPG 檔案
- [ ] 錯誤訊息與確認對話框

### 階段 9：測試與優化（2-3 天）
- [ ] 完整工作流測試
- [ ] 逐幀讀取準確性驗證
- [ ] 效能優化（快取策略、GUI 響應）
- [ ] 錯誤處理完善
- [ ] 使用者提示與說明

### 階段 10：文件與發布（1 天）
- [ ] 使用說明文件
- [ ] 快捷鍵列表
- [ ] 常見問題 FAQ
- [ ] 範例影片與結果

**預計總時程：18-27 天**

---

## 7. 已確認的設計決策

### 7.1 技術細節
1. **右畫布對照邏輯**
   - ✅ 預設 +60 幀（1秒）
   - ✅ 可自訂對照間隔（提供輸入欄位）

2. **Cluster 方向（orientation）**
   - ✅ 自動計算（根據線段標記的 dy 符號）
   - ✅ 顯示確認對話框供使用者驗證
   - ✅ 格式：「cluster_XX: YY mm UP/DOWN, average ZZ mm / 6 frames」

3. **CLAHE 前處理**
   - ✅ 作為可開關的顯示選項
   - ✅ 預設開啟

4. **Inspection 影片**
   - ✅ 停用（不生成）

### 7.2 工作流確認
1. **批次處理**
   - ✅ 單檔專注模式（人工標註不適合批次）

2. **中斷繼續機制**
   - ✅ 載入 CSV 時取得上次位置
   - ✅ 順序讀取至上次位置並繼續

3. **錯誤處理**
   - ✅ 線段標記：每次可清除重繪
   - ✅ Cluster 標記：可刪除整個 Cluster 重新標記

4. **幀導航機制**
   - ✅ 所有導航操作都使用順序讀取
   - ✅ 完全避免 `vidcap.set()` 函數
   - ✅ 包含：載入既有 CSV、所有跳轉（±6/±30/±300）、右畫布對照

### 7.3 使用者體驗
1. **快捷鍵設計**（已確認）
   - **Space**：播放/暫停（最高優先級）
   - **S**：儲存進度（即時儲存）
   - **Z**：復原上一步（Undo）
   - **C**：切換 CLAHE 增強
   - **G**：進入/退出輔助線調整模式
     - 進入時自動暫停播放並顯示輔助線
     - 可拖曳移動輔助線位置
   - **H**：顯示/隱藏輔助線
     - 獨立於調整模式
   - **Q**：退出程式
   - **Enter**：標記 Cluster 起始
   - **Ctrl + Enter**：確認 Cluster 結束
   - **Delete**：刪除當前 Cluster
   - **滑鼠拖曳**：調整模式下拖曳輔助線（±20px 範圍）
   - **注意**：幀導航改用控制面板按鈕（避免與其他功能衝突）

2. **視覺回饋**
   - 線段標記時顯示繪製的線段
   - Cluster 狀態即時更新
   - 進度資訊直接顯示於影像上
   - 按鈕懸停效果（滑鼠移動時高亮）

3. **GUI 架構**
   - ✅ 使用 OpenCV 原生 GUI（cv2.imshow）
   - ✅ 滑鼠回調節流機制（50ms，防崩潰）
   - ✅ 按鈕區域點擊檢測
   - ✅ 快捷鍵快速響應

---

## 8. 預期成果

### 8.1 程式輸出
1. **CSV 檔案**（`lifts/result/21a_dark.csv`）
   - 包含所有關鍵幀的位移資料
   - 人工標記的 cluster 資訊
   - 支援中斷繼續
   - 格式：frame_idx, second, displacement_mm, cluster_id, orientation, frame_path, marking_status

2. **JPG 檔案**（`lifts/exported_frames/21a_dark/`）
   - `pre_cluster_XXX.jpg`：運動起始幀
   - `post_cluster_XXX.jpg`：運動結束幀

3. **Inspection 影片**
   - ❌ 停用

### 8.2 系統特性
- ✅ 可靠的幀讀取（完全避免 OpenCV 導航誤差）
- ✅ 直覺的雙畫布介面
- ✅ 精確的線段標記法（8x 放大 + 三次測量）
- ✅ 自動方向判定與確認
- ✅ 增量式進度儲存
- ✅ 支援中斷繼續（順序讀取至上次位置）
- ✅ 錯誤處理（清除重繪、刪除 Cluster）
- ✅ 單檔專注模式
- ✅ CLAHE 增強可開關

---

## 9. 風險評估與應對

### 9.1 技術風險
- **風險**：順序讀取效能問題（大型影片、向後導航）
  - **應對**：優化快取策略、提供進度指示器
  - **應對**：建議使用者盡量向前導航

- **風險**：OpenCV GUI 響應速度（滑鼠回調頻率過高）
  - **應對**：✅ 滑鼠回調節流機制（50ms 間隔）
  - **應對**：✅ 點擊事件不受節流影響（即時響應）
  - **應對**：✅ 快捷鍵直接響應（無 GUI 延遲）

### 9.2 使用者體驗風險
- **風險**：標記流程過於繁瑣（三次測量）
  - **應對**：設計快捷鍵、優化 ROI 選擇流程
  - **應對**：清晰的進度提示（第 X/3 次測量）

- **風險**：人工標記一致性問題
  - **應對**：三次測量取平均、顯示標準差警告
  - **應對**：提供清除重繪功能

- **風險**：向後導航速度慢（需重新開啟影片）
  - **應對**：提示使用者盡量向前導航
  - **應對**：顯示讀取進度條

---

## 10. 附錄

### 10.1 參考檔案
- `src/manual_correction_tool.py`：線段標記法參考
- `src/lift_travel_detection.py`：主程式比較
- `src/scale_cache_utils.py`：比例尺快取工具
- `src/rotation_utils.py`：旋轉處理工具
- `src/darkroom_utils.py`：暗房區間工具

### 10.2 關鍵參數
- `FRAME_INTERVAL = 6`：關鍵幀間隔
- `DEFAULT_PLAYBACK_DELAY = 150`：預設播放延遲（ms）
- `FAST_PLAYBACK_DELAY = 50`：快速播放延遲（ms）
- `RIGHT_CANVAS_DEFAULT_OFFSET = 60`：右畫布預設偏移（幀）
- `BACKWARD_CACHE_SIZE = 1600`：過去幀快取大小（80%）
- `FORWARD_CACHE_SIZE = 400`：未來幀快取大小（20%）
- `TOTAL_CACHE_MEMORY = 12 GB`：總快取記憶體（2000 關鍵幀）
- `ROI_ZOOM_FACTOR = 8`：線段標記放大倍率
- `STD_WARNING_THRESHOLD = 2.0`：標準差警告閾值（像素）
- `MOUSE_THROTTLE_MS = 50`：滑鼠回調節流間隔（ms）
- `GUIDE_LINE_COLOR = (0, 255, 255)`：輔助線顏色（黃色 BGR）
- `GUIDE_LINE_THICKNESS = 2`：輔助線寬度（像素）

### 10.3 CSV 欄位定義
```python
CSV_COLUMNS = [
    'frame_idx',                      # int: 幀索引
    'second',                         # float: 時間戳（秒）
    'vertical_travel_distance_mm',   # float: 垂直位移（mm，帶符號）
    'cluster_id',                     # int: 群集編號（0=群外）
    'orientation',                    # int: 方向（1=UP, -1=DOWN, 0=群外）
    'frame_path',                     # str: JPG檔名
    'marking_status'                  # str: 'manual' or 'auto'
]
```

---

**計畫書版本**：v3.6 (已確認)
**撰寫日期**：2025-11-11
**狀態**：設計確認完成，準備實作

---

## 11. 變更記錄

### v3.6 (2025-11-11)
- ✅ **重大修正**：線段標記測量方式改為「Y 分量差」
  - 舊方式：測量線段中點的 Y 座標位移（`right_mid_y - left_mid_y`）
  - 新方式：測量線段 Y 分量的變化（`y_right - y_left`）
  - 理由：測量的是垂直結構在 Y 軸方向上的位移
  - 適用場景：精確測量電梯垂直運動
- ✅ **重大修正**：放大倍率從 8x 改為 3x
  - 理由：避免放大後畫面超出螢幕（例如 300x300 ROI → 900x900，並排 1800x900 < 3840 ✅）
  - ROI 尺寸限制：最小 100px，最大 600px
- ✅ **重大改進**：ROI 選擇同步顯示設計
  - 只在左畫布拖曳選擇 ROI
  - 同步在右畫布顯示相同位置的 ROI 矩形框（紅色虛線）
  - 確認後左右畫布同時更新為放大後的 ROI（並排顯示）
  - 左右畫布共用同一個 ROI 區域（避免重複選擇）
- ✅ 更新自動方向判定邏輯
  - 基於線段 Y 分量變化：
    - Δy > 0（Y 分量增加）→ DOWN（向下移動）
    - Δy < 0（Y 分量減少）→ UP（向上移動）
  - 新增詳細註解說明判定邏輯的物理原理與校準建議
- ✅ 更新確認對話框顯示內容：
  - 新增「Y 分量變化」欄位（`Δy=+/-YY.YY px`）
  - 更新輸出日誌格式
- ✅ 更新視覺化範例：
  - ROI 選擇流程（左畫布拖曳，右畫布同步）
  - 3x 放大並排顯示（1800x900）
  - 線段 Y 分量測量示意圖（y_left vs y_right）
- ✅ 更新工作流程說明：
  - 細化線段標記步驟（5.1~5.5）
  - 更新方向判定說明（步驟 6）
  - 更新確認對話框格式（步驟 7）

### v3.5 (2025-11-11)
- ✅ **新增模組**：ClusterManager 類別（協調 CSV 與 JPG 刪除）
  - 確保資料一致性：先查詢 JPG 路徑 → 刪除 JPG → 刪除 CSV
  - 完整刪除：包含運動幀、參考幀、pre/post JPG
  - 錯誤處理：檔案不存在、刪除失敗等情況
  - 使用者回饋：顯示刪除進度與結果
- ✅ 明確 `get_frame_at_offset` 為語法糖（提高可讀性）
  - 加入註解說明：內部直接呼叫 seek_to_frame，避免重複邏輯
- ✅ 技術棧選擇說明：主界面 OpenCV，對話框 tkinter.messagebox
  - 設計原則：效能關鍵路徑使用 OpenCV，非關鍵部分優先考慮開發效率
- ✅ 回應審查意見：
  - 解決快取效能問題（v3.3）
  - 解決 GUI 事件複雜性（v3.4）
  - 解決工具庫依賴矛盾（v3.5）
  - 解決冗餘設計問題（v3.5）

### v3.4 (2025-11-11)
- ✅ **重大改進**：雙畫布改為單一視窗並排顯示（3840x1080）
  - 理由：使用者體驗更好，程式更簡單，效能影響可忽略
  - 輔助線橫跨整個視窗，更清楚顯示運動
  - 視窗管理更簡單（2 個視窗而非 3 個）
- ✅ 簡化滑鼠回調設計：兩個回調（雙畫布、控制面板）
  - 雙畫布回調：處理輔助線拖曳（只關心 y 座標）
  - 控制面板回調：處理按鈕點擊與懸停效果
  - 消除原設計中左右畫布回調的重複邏輯
- ✅ 更新介面視覺化圖表
- ✅ 預留縮放功能（未來如有需要再實作）

### v3.3 (2025-11-11)
- ✅ **重大修正**：快取策略升級為雙向快取（2000 關鍵幀 / 12 GB）
  - backward_cache: 1600 幀（80%）→ 涵蓋過去約 2.7 分鐘
  - forward_cache: 400 幀（20%）→ 涵蓋未來約 40 秒
  - 理由：避免向後導航效能問題
- ✅ **關鍵修正**：統一關鍵幀定義與 JPG 語意
  - 確認關鍵幀序列：0, 6, 12, 18, ...（所有 `frame_idx % 6 == 0` 的幀）
  - 確認 JPG 語意：pre/post 是參考幀（靜止），運動分配到中間
- ✅ 修正等速分配演算法：
  - `end_keyframe = ((end_idx // 6) - 1) * 6`（不包含結束參考幀）
  - 範例修正：1200~1260 的位移分配到 1206~1254（9 幀），每幀 3.333 mm
- ✅ 修正 CSV 結構範例：參考幀的 cluster_id = 0，displacement = 0.0
- ✅ 修正 IncrementalCSVWriter.append_cluster：分別寫入 start/end 參考幀
- ✅ 修正使用者工作流範例中的關鍵幀數量與平均值
- ✅ 更新關鍵參數：BACKWARD_CACHE_SIZE, FORWARD_CACHE_SIZE

### v3.2 (2025-11-10)
- ✅ 輔助線系統設計完成（雙模式：調整模式 + 顯示/隱藏）
  - G 鍵：進入/退出調整模式（可拖曳移動）
  - H 鍵：顯示/隱藏（獨立於調整模式）
  - 拖曳操作：點擊輔助線附近 ±20px 並拖曳
  - 視覺回饋：調整模式虛線，正常模式實線
  - 自動暫停：進入調整模式時自動暫停播放
  - 左右同步：兩個畫布共用同一 Y 座標

### v3.1 (2025-11-10)
- ✅ 新增輔助線系統基礎設計（使用方向鍵調整）
- ✅ 新增 GuideLineSystem 類別規劃

### v3.0 (2025-11-10)
- ✅ **重大變更**：GUI 架構從 Tkinter 改為 OpenCV 原生 GUI
  - 理由：提升響應速度，降低系統負載
  - 使用 cv2.imshow() + 滑鼠回調 + 鍵盤快捷鍵
  - 直接在影像上繪製按鈕和文字
- ✅ 新增滑鼠回調節流機制（50ms，防止崩潰）
- ✅ 新增按鈕區域點擊檢測實作
- ✅ 強化快捷鍵系統設計（Space, S, Z, C, Q 等）
- ✅ 更新階段 3 實作細節（OpenCV GUI 播放器）
- ✅ 新增關鍵參數：MOUSE_THROTTLE_MS
- ✅ 更新風險評估（OpenCV GUI 響應速度）

### v2.0 (2025-11-10)
- ✅ 確認右畫布對照間隔可自訂（預設 60 幀）
- ✅ 確認自動方向判定機制與確認對話框格式
- ✅ 確認 CLAHE 作為可開關選項
- ✅ 確認停用 inspection 影片
- ✅ 確認單檔專注模式
- ✅ 確認錯誤處理機制（清除重繪 + 刪除 Cluster）
- ✅ 強化逐幀讀取機制說明（所有導航都使用順序讀取）
- ✅ 新增快捷鍵設計
- ✅ 新增詳細的 CSV 欄位定義

### v1.0 (2025-11-10)
- 初版計畫書
