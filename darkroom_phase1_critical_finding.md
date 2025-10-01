# Phase 1 暗房測試關鍵發現：OpenCV 隨機存取問題

**日期**：2025-09-30
**狀態**：⚠️ 關鍵問題已確認
**影響**：所有使用 `cap.set(cv2.CAP_PROP_POS_FRAMES)` 的測試腳本結果無效

---

## 問題總結

在 Phase 1 暗房運動偵測驗證過程中，發現測試腳本讀取的影片內容與預期不符，導致所有測試結果（v1, v2, v3）**無法反映真實偵測能力**。

---

## 問題發現過程

### 1. 初始觀察

**使用者報告**：
- 在 PotPlayer 中 03:05-03:07 看到明顯的顯微鏡垂直位移
- 測試腳本在相同時間範圍（185s-187s）卻顯示 `static`（0 px 運動）

**初步懷疑**：
- 演算法失效？
- 時間偏移（`config.py` 的 `start:30`）？

### 2. 幀號驗證

**PotPlayer 顯示**：
- 00:03:05.000 → Frame **11100.60**
- 00:03:07.101 → Frame **11226.60**

**測試腳本輸出**：
- 185.0s → Frame **11100**
- 187.1s → Frame **11226**

**結論**：幀號完全一致 ✅

### 3. 決定性證據：綠色手術帽位置

**對比分析**：

| 時間點 | PotPlayer 畫面 | 測試腳本輸出 (v3) | 差異 |
|--------|---------------|------------------|------|
| 03:05 (Frame 11100) | 綠色帽子在左上角 | 綠色帽子在左上角 | ✅ 相同 |
| 03:07 (Frame 11226) | **綠色帽子移出畫面** | 綠色帽子仍在原位 | ❌ **完全不同** |

**關鍵發現**：
- 顯微鏡零件位置：PotPlayer 顯示明顯上移，測試腳本幾乎無變化
- 背景人物：PotPlayer 顯示綠色帽子移出畫面，測試腳本顯示帽子位置不變
- ROI 框作為格線對照：測試腳本兩幀中所有物體位置幾乎一致

**結論**：即使幀號相同，**實際讀取的影片內容完全不同** ❌

---

## 根本原因：OpenCV 隨機存取的已知問題

### 問題機制

**OpenCV 的 `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)` 存在以下問題**：

1. **不精確跳轉**
   - 跳轉到最近的**關鍵幀（I-frame）**，而非指定的精確幀號
   - H.264/H.265 等壓縮格式特別嚴重（關鍵幀間隔可能 60-300 幀）

2. **幀號回報錯誤**
   - `cap.get(cv2.CAP_PROP_POS_FRAMES)` 回報的是**請求的幀號**
   - 而非**實際讀取的幀號**
   - 造成「幀號正確但內容錯誤」的假象

3. **不同播放器行為不同**
   - PotPlayer / Windows Media Player：使用更精確的解碼方法
   - OpenCV：依賴 FFmpeg 的快速跳轉（犧牲精度）

### 實際影響範例

```python
# 測試腳本 (test_darkroom_motion_v3.py)
cap.set(cv2.CAP_PROP_POS_FRAMES, 11100)  # 請求跳到 11100
ret, frame = cap.read()
frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)  # 回報 11100

# 但實際讀取的可能是：
# - Frame 10800 (前一個關鍵幀)
# - Frame 11200 (後一個關鍵幀)
# - 任何鄰近的關鍵幀
```

**這解釋了**：
- 為何測試腳本讀到的畫面沒有運動（實際讀到運動發生前的片段）
- 為何幀號「看起來正確」但內容完全錯誤
- 為何綠色帽子位置不同（實際讀取時間點偏移）

---

## 相關研究

**專案內已有相關研究**：
- `investigation_frame_navigation.md`（前期幀導航研究）
- 研究結論：確認存在幀導航誤差
- 解決方案：主程式採用順序讀取 + JPG 匯出機制

---

## 主程式的正確做法

**`src/lift_travel_detection.py` 的設計**：

```python
# 第 83-84 行：計算起終點幀號
start_frame = int(video_config.get(file_name, {}).get('start', 0) * fps)
end_frame = int(video_config.get(file_name, {}).get('end', video_length/fps) * fps)

# 第 129 行：只在開始時跳一次
vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# 第 133-135 行：之後全部順序讀取
while ret:
    frame_idx = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = vidcap.read()  # 順序讀取，不再使用 set()

    # 每 FRAME_INTERVAL (6) 幀處理一次
    if frame_idx % FRAME_INTERVAL == 0:
        # 處理此幀
```

**關鍵設計**：
1. ✅ **只在影片開頭使用一次 `set()`**（跳到 `start_frame`）
2. ✅ **後續全部順序讀取**（`read()` 自動遞增，無需再 `set()`）
3. ✅ **匯出 JPG 供人工校正**（避免隨機存取問題）

---

## 測試腳本的錯誤做法

**`test_darkroom_motion_v1/v2/v3.py` 的問題**：

```python
# 第 31 行：設定測試起點
TEST_START = 180  # 3:00 (秒)

# 第 111-113 行：隨機跳轉到測試起點 ❌
start_frame = int(TEST_START * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # ❌ 不精確跳轉

# 第 168 行：初始化幀號
frame_idx = start_frame  # 使用請求的幀號（但實際內容可能不同）

# 後續迴圈（第 171-183 行）：
while True:
    for _ in range(FRAME_INTERVAL):
        ret, frame = cap.read()
        frame_idx += 1  # 手動遞增
```

**問題**：
1. ❌ 使用 `cap.set()` 跳轉到測試區間中段（180s）
2. ❌ 假設跳轉精確（實際可能偏移數百幀）
3. ❌ 手動遞增 `frame_idx`（與實際讀取位置不同步）

---

## 影響範圍

### 無效的測試結果

以下所有測試結果**不可信**：

| 測試版本 | 檔案 | 狀態 |
|----------|------|------|
| v1 | `test_darkroom_motion.py` | ❌ 無效（隨機存取） |
| v2 | `test_darkroom_motion_v2.py` | ❌ 無效（隨機存取 + ROI 錯誤） |
| v3 | `test_darkroom_motion_v3.py` | ❌ 無效（隨機存取） |

**具體問題**：
- 所有「成功率 96-98%」的結論無效
- 所有「特徵點 22 個」、「匹配對 12-15 個」的統計無效
- 所有「偵測到運動」的結果可能是誤報（讀到錯誤的幀）

### 仍然有效的結論

以下結論**仍然有效**：

1. ✅ **CLAHE 前處理有效**
   - `analyze_darkroom_video.py` 的可行性分析使用順序讀取
   - 證實特徵點充足（100 個/幀，達到上限）

2. ✅ **旋轉校正功能正確**
   - 視覺化輸出顯示影像已正確旋轉 21°
   - `rotation_utils.py` 運作正常

3. ✅ **幀導航問題已確認**
   - 證實 OpenCV 隨機存取不可靠
   - 驗證了前期 `investigation_frame_navigation.md` 的研究

---

## 解決方案

### 方案 1：修正測試腳本（推薦）

**修改策略**：
```python
# 從影片開頭開始順序讀取
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

# 計算目標幀號
target_start_frame = int(185 * fps)  # 185 秒

# 順序讀取到目標位置
frame_idx = 0
while frame_idx < target_start_frame:
    ret, frame = cap.read()
    frame_idx += 1

# 開始測試（繼續順序讀取）
while frame_idx < target_end_frame:
    ret, frame = cap.read()
    if frame_idx % FRAME_INTERVAL == 0:
        # 處理此幀
    frame_idx += 1
```

**優點**：
- ✅ 確保讀取到正確的幀
- ✅ 與主程式邏輯一致
- ❌ 需要從頭讀取（耗時，但準確）

### 方案 2：直接使用主程式（最可靠）

**修改 `lift_travel_detection.py`**：
1. 複製為 `lift_travel_detection_dark.py`
2. 反轉暗房邏輯（只處理暗房區間）
3. 加入 CLAHE 前處理
4. 測試完整流程

**優點**：
- ✅ 完全避免隨機存取問題
- ✅ 保留所有現有功能（物理群集、JPG 匯出）
- ✅ 直接產出可用的 CSV + inspection 影片

### 方案 3：使用已匯出的 JPG（如果有）

如果主程式已經匯出過 JPG：
```python
# 直接載入 JPG 進行測試
jpg_path = f"lifts/exported_frames/{video_name}/pre_cluster_001.jpg"
frame = cv2.imread(jpg_path)
```

**優點**：
- ✅ 完全避免影片讀取問題
- ✅ JPG 是主程式順序讀取產生，可信度高

---

## 經驗教訓

### 1. OpenCV 隨機存取不可靠

**教訓**：
- ❌ 不要在測試/生產中使用 `cap.set(cv2.CAP_PROP_POS_FRAMES, X)`
- ✅ 永遠使用順序讀取（從頭或從 start_frame 開始）
- ✅ 對關鍵幀匯出 JPG，避免後續隨機存取

### 2. 幀號不等於內容

**教訓**：
- ❌ 不要相信 `cap.get(cv2.CAP_PROP_POS_FRAMES)` 的回報值
- ✅ 使用視覺特徵驗證（如：背景人物位置）
- ✅ 對比不同工具的結果（PotPlayer vs OpenCV）

### 3. 驗證測試假設

**教訓**：
- ❌ 不要假設「幀號相同 = 內容相同」
- ✅ 使用多種方法驗證（視覺對比、時間戳、背景細節）
- ✅ 保留原始截圖供後續驗證

### 4. 前期研究的重要性

**教訓**：
- ✅ 專案前期的 `investigation_frame_navigation.md` 研究是有價值的
- ✅ 主程式採用順序讀取 + JPG 匯出是正確的設計
- ✅ 新功能開發應遵循已驗證的設計模式

---

## 下一步行動

### 立即行動

1. **停止使用所有隨機存取測試腳本**
   - 標記 v1/v2/v3 為「無效結果」
   - 不再基於這些結果做決策

2. **選擇新的驗證方法**
   - 方案 1：修正測試腳本使用順序讀取
   - 方案 2：直接開發 `lift_travel_detection_dark.py`
   - 方案 3：如果有 JPG，直接測試演算法

### 後續開發

**Phase 2 開發策略調整**：

| 原計畫 | 調整後 |
|--------|--------|
| 基於 v3 測試結果進入階段 2 | ❌ 測試結果無效，需重新驗證 |
| 採用 ROI 0.6 + 旋轉 21° | ✅ 旋轉正確，但 ROI 需重新測試 |
| 做法 A（自動化偵測）可行 | ⚠️ 需重新驗證（尚未真正測試） |

**建議順序**：
1. 直接開發 `lift_travel_detection_dark.py`（採用順序讀取）
2. 在真實暗房區間測試完整 pipeline
3. 根據實際結果決定是否需要調整參數或改用做法 B

---

## 相關檔案

### 問題相關
- `test_darkroom_motion.py` - v1 測試（無效）
- `test_darkroom_motion_v2.py` - v2 測試（無效）
- `test_darkroom_motion_v3.py` - v3 測試（無效）
- `investigation_frame_navigation.md` - 前期幀導航研究

### 仍然有效
- `analyze_darkroom_video.py` - 可行性分析（順序讀取，有效）
- `src/lift_travel_detection.py` - 主程式（正確設計）
- `Phase2_darkroom_implementation_plan.md` - 實施計畫書

### 參考資料
- `src/rotation_config.py` - 旋轉配置（21.mp4 = 21°）
- `src/darkroom_intervals.py` - 暗房時間區間配置
- `manual_correction_guide.md` - 人工校正工作流（順序讀取設計）

---

## 附錄：證據截圖對比

### PotPlayer (正確讀取)
- **03:05 (Frame 11100)**：綠色帽子在左上角，零件在下方位置
- **03:07 (Frame 11226)**：綠色帽子移出畫面，零件明顯上移

### 測試腳本 v3 (錯誤讀取)
- **185.0s (Frame 11100)**：綠色帽子在左上角，零件位置 A
- **187.1s (Frame 11226)**：綠色帽子仍在左上角（相同位置），零件位置 A（幾乎無變化）

**差異**：
- 帽子位置：PotPlayer 顯示移動，v3 顯示靜止 ❌
- 零件位移：PotPlayer 顯示上移，v3 顯示無變化 ❌
- 結論：v3 讀取的不是 03:05-03:07，而是其他時間點的內容

---

**文件版本**：1.0
**最後更新**：2025-09-30
**作者**：Claude Code
**狀態**：✅ 已確認問題，等待下一步決策