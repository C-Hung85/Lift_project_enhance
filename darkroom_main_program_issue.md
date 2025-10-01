# 暗房主程式 (lift_travel_detection_dark.py) 審視報告

**日期**: 2025-10-01
**審視目標**: 根據 `darkroom_main_program_implementation.md` 實施計畫，評估 `src/lift_travel_detection_dark.py` 的實作完成度與潛在風險。
**更新日期**: 2025-10-01
**處理狀態**: ✅ 已修正風險 1 和風險 3，澄清風險 2

---

## 總體評價

`lift_travel_detection_dark.py` 的實作完成度很高，絕大部分都遵循了實施計畫中的規範。程式碼結構清晰，並且成功地反轉了主程式的偵測邏輯，使其專注於暗房區間。

在審視過程中，發現了 **1 個重大風險**（已修正）、**1 個規格理解錯誤**（已澄清），以及 **1 個次要的顯示問題**（已修正）。

---

## 主要發現與處理結果

以下為識別出的主要風險及處理狀態，按嚴重性排序：

### 1. ✅ (重大) 比例尺查詢錯誤導致位移計算不準確 - 已修正

- **問題描述**:
  在 `lift_travel_detection_dark.py` 的 `scan` 函數中，將像素位移 `delta_px` 轉換為物理位移 `mm` 時，程式碼使用 `file_name` 來查詢比例尺因子 `scale_factor`。根據實施計畫，所有參數查詢都應使用 `base_name` (例如 `'21.mp4'`)，但這裡卻用了 `file_name` (例如 `'21a.mp4'`)。

  原始錯誤位置（三處）：
  - Line 343: `scale_factor = video_scale_dict.get(file_name, 1.0)`
  - Line 372: `effective_mm_current = delta_px * 10 / video_scale_dict.get(file_name, 1.0)`
  - Line 414-418: 比例尺查詢區塊

- **潛在影響**:
  此錯誤將導致程式在 `video_scale_dict` 中找不到暗房影片（如 `21a.mp4`）的比例尺資料，從而退回使用預設值 `1.0`。這會造成 CSV 輸出中 `vertical_travel_distance (mm)` 欄位的計算結果**完全錯誤**。

- **修正內容** (2025-10-01):
  已將所有三處 `video_scale_dict.get(file_name, ...)` 修改為 `video_scale_dict.get(base_name, ...)`：

  ```python
  # Line 343: 回填 pending 幀的比例尺計算
  scale_factor = video_scale_dict.get(base_name, 1.0)

  # Line 372: InCluster 狀態的有效位移計算
  effective_mm_current = delta_px * 10 / video_scale_dict.get(base_name, 1.0)

  # Line 414-418: 比例尺查詢區塊（已使用 base_name，警告訊息增強）
  if base_name in video_scale_dict:
      scale_factor = video_scale_dict[base_name]
  else:
      print(f"⚠️  警告: 影片 {file_name} (base: {base_name}) 沒有有效的比例尺資料，使用預設值 1.0")
  ```

- **驗證狀態**: ✅ 已完成修正，三處查詢均已統一使用 `base_name`

### 2. ✅ (高) 運動方向定義誤解 - 已澄清（無需修正）

- **原始問題描述**:
  實施計畫 `8.3 已確認決策` 中明確規定：「**向上移動 = 正值**」。
  報告原本認為程式碼中的位移計算 `kp2_coord[1]-kp1_coord[1]` 在 OpenCV 座標系下（y軸向下），向上移動（y值減小）會產生**負值**。

- **澄清與正確理解** (2025-10-01):

  **此風險評估是錯誤的**。程式邏輯實際上已經正確實現「向上為正，向下為負」的規格。

  **關鍵機制**：K-means 分組與符號反轉邏輯（line 189-192 / 245-248）

  ```python
  # lift_travel_detection.py 與 lift_travel_detection_dark.py 都有相同邏輯
  if abs(group0_v_travel) > abs(group1_v_travel):
      vertical_travel_distance = int(group1_v_travel - group0_v_travel)  # 較小組減較大組
  else:
      vertical_travel_distance = int(group0_v_travel - group1_v_travel)  # 較小組減較大組
  ```

  **運作原理**：
  1. OpenCV 原始座標：`kp2[1] - kp1[1]` 計算（向下為正，向上為負）
  2. K-means 分出「移動物體」與「靜止背景」兩組
  3. **關鍵反轉**：程式使用「較小組 - 較大組」，實現符號反轉：
     - 電梯向下：原始 +50 → group0=+50, group1=0 → `0 - 50 = -50` ✅ (向下為負)
     - 電梯向上：原始 -50 → group0=-50, group1=0 → `0 - (-50) = +50` ✅ (向上為正)

  **實際驗證**：透過 inspection video 截圖驗證：
  - 物體向下移動時，顯示 `-2.08817 mm` (負值) ✅
  - 符合使用者直覺預期：向上為正，向下為負

- **結論**:
  - ❌ 原報告對此風險的分析是錯誤的
  - ✅ 程式邏輯完全正確，已實現規格要求
  - ✅ 兩個程式 (`lift_travel_detection.py` 和 `lift_travel_detection_dark.py`) 邏輯一致
  - **無需任何修正**

### 3. ✅ (次要) Inspection Video 顯示文字誤導 - 已修正

- **問題描述**:
  在 `lift_travel_detection_dark.py` 產生的 inspection video 中，當進入暗房區間時，畫面會顯示文字 `"darkroom (ignored)"`。這段文字是從原主程式繼承而來的。

- **潛在影響**:
  在此暗房專用腳本中，暗房區間是**唯一被處理**的區間，而非被忽略。這段文字會對觀看影片的人產生誤導，使其誤以為暗房中的運動沒有被計算。

- **修正內容** (2025-10-01):
  已將 line 466 的顯示文字從 `"darkroom (ignored)"` 修改為 `"darkroom (active)"`：

  ```python
  # Line 466: 修正後
  if is_darkroom:
      display_text = "darkroom (active)"
      text_color = (128, 128, 128)  # 灰色
  ```

- **驗證狀態**: ✅ 已完成修正，inspection video 將正確顯示暗房區間處理狀態

---

## 最終結論

### 修正摘要

| 風險項目 | 嚴重程度 | 原始評估 | 實際狀況 | 處理結果 |
|---------|---------|---------|---------|---------|
| **風險 1：比例尺查詢錯誤** | 🔴 重大 | ✅ 正確 | 重大錯誤 | ✅ **已修正** (三處) |
| **風險 2：運動方向定義** | 🟡 高 | ❌ 錯誤 | 程式正確 | ✅ **已澄清** (無需修正) |
| **風險 3：顯示文字誤導** | 🟢 次要 | ✅ 正確 | 次要問題 | ✅ **已修正** (一處) |

### 程式狀態

✅ `lift_travel_detection_dark.py` 已完成所有必要修正，可以進行測試與部署：

1. **風險 1 (比例尺查詢)** - 已修正
   - Line 343: 使用 `base_name` 查詢比例尺
   - Line 372: 使用 `base_name` 計算有效位移
   - Line 414-418: 比例尺查詢邏輯已正確，警告訊息增強

2. **風險 2 (運動方向)** - 無需修正
   - 程式邏輯透過 K-means 分組實現正確的符號反轉
   - 實際行為符合規格：向上為正，向下為負
   - 與主程式 `lift_travel_detection.py` 邏輯完全一致

3. **風險 3 (顯示文字)** - 已修正
   - Line 466: 顯示文字改為 `"darkroom (active)"`
   - 正確反映暗房區間為處理對象而非忽略對象

### 建議後續行動

1. ✅ 可以開始對暗房影片進行完整測試
2. ✅ 驗證比例尺計算的正確性（特別是 `21a.mp4` 等暗房影片）
3. ✅ 檢查 inspection video 中的顯示文字是否正確更新
4. 📋 若測試通過，可進行資料融合與分析工作
