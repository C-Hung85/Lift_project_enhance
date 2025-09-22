## 調查報告：手動校正工作流之「影片幀導航不準」議題

### 1) 現象摘要
- **使用情境**: 針對 `lifts/result/c1.csv` 的第一個非 0 位移群集進行人工校正。
- **期待**: 依據 CSV 群集的位移總和與比例尺，預期兩條線段的 Y 分量應有約 7 像素以上的差距。
- **實際觀察**: 工具在兩個時點（群集前零點 vs 群集末點）顯示的 ROI 幾乎重合，兩條線段的 Y 分量差僅 0–1 像素。
- **已排除**:
  - 以 `second * fps` 估算 vs 使用 `frame_idx` 精確導引，結果無差異。
  - 放大倍率與人工繪製能力不足之說（在外部播放器與小畫家可清楚辨識 ~10 px 級位移）。
  - 設備本身確實有伸縮位移（非演算法誤解）。

結論（暫定）：不是使用者標註精度問題，而是兩個時點顯示的幀內容「高度相似」，導致線段差異遠小於預期。

---

### 2) 重點日誌與程式行為片段
以下節錄有助於聚焦問題位置（僅做佐證，非修正建議）。

```486:546:src/manual_correction_tool.py
        print(f"\n=== 時戳/幀號調試信息 ===")
        print(f"當前階段: {self.current_phase}")
        ...
        if frame_id is not None and self.data_manager.use_frame_indices:
            frame = self.video_handler.get_frame_at_index(frame_id)
            print(f"使用幀號 {frame_id} 進行精確定位")
        else:
            frame = self.video_handler.get_frame_at_timestamp(timestamp)
            print(f"退回使用時戳 {timestamp:.3f}s 進行估算定位")
        ...
        self.show_frame(frame)
```

```302:365:src/manual_correction_tool.py
    def get_frame_at_index(self, frame_number) -> Optional[np.ndarray]:
        ...
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        ...
        ret, frame = self.cap.read()
        ...
        if self.rotation_angle != 0:
            frame = rotate_frame(frame, self.rotation_angle)
        return frame
```

```675:693:src/manual_correction_tool.py
    def enter_precision_marking_mode(self):
        roi_x, roi_y, roi_w, roi_h = self.roi_rect
        roi_frame = self.original_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        enlarged_roi = cv2.resize(roi_frame, None, fx=self.zoom_factor, fy=self.zoom_factor, ...)
        self.display_frame_only(enlarged_roi)  # 僅更新畫面顯示，不覆寫 original_frame
```

```724:763:src/manual_correction_tool.py
    def display_frame_only(self, frame: np.ndarray):
        # 產生 PhotoImage 並繪到 canvas；更新 self.image_bounds，但不更新 self.original_frame
```

```764:793:src/manual_correction_tool.py
    def place_line_point(...):
        # 反推座標：canvas → (放大ROI座標) → 原影像座標
        roi_local_x = int((canvas_x - img_x) / self.display_scale)
        roi_local_y = int((canvas_y - img_y) / self.display_scale)
        original_x = roi_x + (roi_local_x // self.zoom_factor)
        original_y = roi_y + (roi_local_y // self.zoom_factor)
```

```895:915:src/manual_correction_tool.py
    def pixel_to_canvas_coords(self, pixel_coords):
        # 正向映射：原影像座標 → canvas
        canvas_x = img_x + (local_x * self.zoom_factor * self.display_scale)
        canvas_y = img_y + (local_y * self.zoom_factor * self.display_scale)
```

附帶指出一個與「導航」無直接關係但確實存在的錯誤，後續應修（非本輪處理）：

```224:263:src/manual_correction_tool.py
    def apply_correction(...):
        displacement_col = self.df.columns[1]
        # 對於三欄 CSV（frame_idx, second, displacement），此處 columns[1] 是 'second'，
        # 會把更正值寫到時間欄而非位移欄。正確應使用位移欄（索引依 CSV 結構判定）。
```

---

### 3) 可能成因（聚焦「幀導航/顯示」管線），依優先級排列
1. 影片來源不一致（高可能/易驗證）
   - 工具固定從 `lifts/data/1.mp4` 讀取；人工比對使用的播放器檔案若為 `lifts/inspection/1.mp4` 或其他來源，兩者內容可能不同或前處理不同（旋轉、剪裁、重編碼）。
   - 建議先核對 console 中「影片檔案」路徑與外部播放器是否完全一致（含資料夾）。

2. 群集端點的「位移峰值並未發生於群集尾端」而是群集中途（高可能/設計議題）
   - 目前工具取樣點為「前零點」與「群集末點」。群集總位移（sum of abs）≈ 0.812 mm ≈ 7 px，並不保證「末點」就是離「前零點」最遠的瞬間。
   - 若峰值出現在群集中段，前零點 vs 末點只會看到部分位移甚至極小差值。
   - 這屬工作流設計而非導引錯誤，但對使用者體驗會被感知為「導航到不對的幀」。

3. OpenCV 尋幀在特定 MP4（B-frames/VFR）上的邊界行為（中可能/需實證）
   - 雖日誌顯示 `set` 與 `read` 成功，實務上少數編碼器/封裝會導致 seek 後的第一張圖像不是期望幀（或幀時間分佈不均）。
   - 需透過「直接匯出 frame 6852 與 6876 為 PNG」來與 GUI 顯示比對，以排除編碼器層面的異常。

4. ROI/座標映射鏈中的整數量化（低至中可能/微幅影響）
   - 逆向映射採用兩次整數化：`int(/display_scale)` 後再 `// zoom_factor`。理論上可能造成 ±1px 誤差，但不至於把 ~7px 壓成 0–1px。
   - 仍建議在後續修正階段保留浮點到最後一步再取整，以降低量化誤差。

5. 使用者標記策略與視覺參考（中性因素）
   - 線段抗抖動假設其中一端或兩端對應的是「同一結構特徵」。若兩端都跟隨會同向移動的邊緣，線段長度可能變化較小。
   - 建議固定一端在穩定背景/固定結構，另一端貼伸縮端，避免「兩端一起飄」。

---

### 4) 不改碼驗證步驟（建議先做，快速收斂問題）
- 核對檔來源一致性（必做）
  - 直接比對 console 輸出：`影片檔案: lifts/data/1.mp4` 是否與外部播放器開啟的檔案完全一致（含資料夾）。

- 匯出幀影像交叉驗證（必做）
  - 以外部腳本（或 ffmpeg）從 `lifts/data/1.mp4` 匯出 frame 6852 與 6876 的 PNG，肉眼比較是否存在預期位移差。
  - 再把 GUI 中兩階段的 ROI 畫面擷取（目前可用系統截圖）與匯出的原幀相互比對，確認 GUI 顯示的確就是這兩張幀。

- 群集端點 vs 峰值檢核（強烈建議）
  - 先不改程式，於 CSV 中臨時把該群集的「末幀」`frame_idx` 手動向後偏移 +6、+12 幀（只做觀察），再啟動工具觀察線段差是否明顯放大；若是，問題屬「取點策略」而非導引錯誤。

- Windows 顯示縮放與視窗尺寸檢核（可選）
  - 將 Windows 顯示縮放設為 100%，並最大化 GUI 視窗，重做一次觀察，排除視覺縮放造成的錯覺或取點困難。

---

### 5) 之後的修正方向（本輪不動手，供討論）
- 幀微調快捷鍵：在第二條線段時提供「←/→ = ±1 幀；Shift + ←/→ = ±5 幀」的微調，便於在群集尾附近找到位移峰值幀。
- 幀對幀對比輔助：切到第二條線段時，支援「差分/閃爍/半透明疊圖」顯示第一幀 ROI，協助目視辨識位移。
- 一鍵導出比對材料：把兩個幀（含 ROI）直接輸出為 PNG 與日誌捆綁 frame_idx，便於離線比對與回報。
- 降低量化誤差：座標反推保留浮點到最後一步再取整，減少 ±1px 的量化影響。
- 修正資料欄寫入：修掉 `apply_correction` 在三欄 CSV 時寫到 `second` 欄的問題（與本議題無關，但會影響資料正確性）。

---

### 6) 建議的下一步討論
1. 先完成「來源一致性」與「幀匯出」兩項驗證，確認 GUI 顯示的 6852/6876 幀是否與外部播放器看到的相同內容。
2. 若兩張幀確實內容高度相似，請再嘗試把末幀臨時手動調為 `6876 + 6` 或 `+12` 重新觀察；若差異立刻達到 7–10 px，則可判定為「取點策略」而非導引錯誤。
3. 取得這兩張幀的對比截圖後，我們再決定要優先加入「幀微調」與「差分疊圖」哪一項以最快改善使用體驗。

---

若有需要，我可以先準備一個極簡外部腳本，專門把指定 `frame_idx` 的幀擷取成 PNG（不動現有 GUI 與邏輯），供你快速核對。亦可依你提供的幀號清單，一次匯出多張對比幀。


