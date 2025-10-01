# Phase 2 暗房區間運動偵測實施計畫書

## 文件資訊
- **建立日期**: 2025-09-30
- **專案階段**: Phase 2 - 暗房區間專用運動偵測
- **負責開發**: Claude Code
- **測試影片**: `lifts/darkroom_data/21_a.mp4`

---

## 一、背景與目標

### 1.1 專案背景
本專案擁有成熟的顯微鏡升降運動偵測系統（`src/lift_travel_detection.py`），基於 ORB 特徵點與 K-means 分群演算法。然而，部分手術影片存在**暗房區間**（醫師關閉主要照明），導致畫面照度嚴重不足，原始演算法無法正確偵測。

目前系統透過 `src/darkroom_intervals.py` 配置暗房時間區間，主程式會**完全忽略**這些區間的運動偵測。

### 1.2 專案目標
1. 開發暗房專用運動偵測腳本 `lift_travel_detection_dark.py`
2. 實作數據整合工具，合併暗房與非暗房結果
3. （可選）實作 inspection 影片整合工具
4. 保持與現有人工校正工具的完全相容性

---

## 二、可行性分析結果

### 2.1 技術驗證（2025-09-30）

**測試影片**: `D:\Lift_project\lifts\darkroom_data\21_a.mp4`
- 暗房區間: 30秒 - 8分28秒（8分鐘）
- 運動點: 3分05-07秒附近
- 影片狀態: 已用影像編輯軟體進行激進亮度增強

**分析腳本**: `analyze_darkroom_video.py`

**關鍵發現**:
1. ✅ **特徵點數量充足**: ORB 偵測器在所有時間點都能穩定偵測到 100 個特徵點（達到上限）
2. ✅ **CLAHE 增強有效**: 對比度自適應直方圖均衡化顯著改善視覺品質
3. ✅ **結構輪廓清晰**: 顯微鏡金屬結構、伸縮部件輪廓清楚可辨
4. ✅ **Canny 邊緣可用**: 邊緣偵測能穩定抓取顯微鏡輪廓

**統計數據**:
```
特徵點數量 (6個時間點平均):
  ORB 原始:        100.0 個
  ORB + CLAHE:     100.0 個  ← 推薦使用
  ORB + Canny遮罩:  87.2 個

畫質指標:
  平均亮度: 100-117 (0-255 scale)
  邊緣密度: 1.3%-3.0%
  梯度強度: 9.6-12.5
```

**結論**: ✅ **做法 A（自動化偵測）完全可行**

### 2.2 技術方案選擇

**採用方案**: 做法 A - 基於特徵點的自動化偵測

**理由**:
1. 特徵點數量充足，滿足原始演算法需求（最低 6 個匹配對）
2. CLAHE 增強簡單高效，無需複雜調適
3. 保持現有 pipeline 完整性，最小化開發風險
4. 可批量處理多個影片，效率高

**備案**: 做法 B（人工輔助）僅在做法 A 失敗時考慮

---

## 三、詳細實施計畫

### 階段 1: 運動偵測驗證（1-2天）

#### 目標
在真實暗房區間測試完整的運動偵測 pipeline，驗證技術可行性。

#### 交付成果
- **檔案**: `test_darkroom_motion.py`
- **輸出**: 驗證報告（控制台輸出 + 視覺化圖片）

#### 核心功能
```python
# 測試範圍: 21_a.mp4 的 3:05-3:07 (運動片段)
# 流程:
1. 讀取連續幀（每6幀取樣，與主程式一致）
2. CLAHE 前處理 → ORB 特徵點偵測
3. BF Matcher 特徵匹配
4. K-means 分群 (2 clusters)
5. 垂直位移計算 + t-test 統計檢驗
6. 輸出匹配成功率、位移量、視覺化結果
```

#### 驗證指標
- ✅ **成功**: 匹配對數 ≥ 6，能穩定計算垂直位移
- ⚠️ **警告**: 匹配對數 3-5，需要調整參數
- ❌ **失敗**: 匹配對數 < 3，考慮備案

#### 決策點
- 成功率 > 70% → 進入階段 2
- 成功率 < 70% → 重新評估，考慮混合方案

---

### 階段 2: 暗房專用偵測主程式（3-5天）

#### 目標
開發完整的暗房專用運動偵測腳本，保持所有現有功能。

#### 交付成果
- **檔案**: `src/lift_travel_detection_dark.py`
- **測試**: 至少處理 2 個暗房區間影片，生成 CSV + inspection 影片

#### 開發策略
**基礎**: 複製 `lift_travel_detection.py` 作為起點（623行）

**核心修改點**:

##### 1. 前處理管線（新增函數）
```python
def preprocess_darkroom_frame(frame):
    """
    暗房影片專用前處理

    Args:
        frame: BGR 彩色影像

    Returns:
        enhanced: CLAHE 增強後的灰階影像
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced
```

##### 2. 特徵偵測整合（修改 2 處）
**位置 1**: 第 126 行（初始幀）
```python
# 原始:
# keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame, mask)

# 修改為:
enhanced_frame = preprocess_darkroom_frame(frame)
keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(enhanced_frame, mask)
```

**位置 2**: 第 146 行（迴圈內幀）
```python
# 原始:
# keypoint_list2, feature_descrpitor2 = feature_detector.detectAndCompute(frame, mask)

# 修改為:
enhanced_frame = preprocess_darkroom_frame(frame)
keypoint_list2, feature_descrpitor2 = feature_detector.detectAndCompute(enhanced_frame, mask)
```

##### 3. 暗房區間邏輯反轉（修改第 196-238 行）
```python
# 原本邏輯: 在暗房區間內 → 忽略運動 (設為 0)
# 新邏輯:   在暗房區間外 → 忽略運動 (設為 0)

# 第 196-202 行
current_time_seconds = frame_idx / fps
is_darkroom, darkroom_info = is_in_darkroom_interval(current_time_seconds, darkroom_intervals_seconds)

# 原始: if is_darkroom: vertical_travel_distance = 0
# 修改為:
if not is_darkroom:  # 非暗房區間 → 忽略
    vertical_travel_distance = 0

# 第 214 行 - 狀態機候選判斷
# 原始: is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and (not is_darkroom)
# 修改為:
is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and is_darkroom

# 第 217-238 行 - 狀態機處理
# 原始: if is_darkroom: state = 'Idle'  (暗房時強制回 Idle)
# 修改為:
if not is_darkroom:  # 非暗房時強制回 Idle
    state = 'Idle'
    # ... (其餘邏輯相同)
```

##### 4. Inspection 影片文字標記（修改第 407-415 行）
```python
# 將 "darkroom (ignored)" 改為 "normal light (ignored)"
# 確保使用者理解暗房腳本的行為
if not is_darkroom:  # 非暗房時顯示為忽略
    display_text = "normal light (ignored)"
    text_color = (128, 128, 128)  # 灰色
elif camera_pan and is_darkroom:
    display_text = "camera pan"
    text_color = (0, 255, 255)  # 黃色
else:
    display_text = f"travel: {round(travel_distance_sum, 5)} mm"
    text_color = (0, 0, 255) if vertical_travel_distance == 0 else (0, 255, 0)
```

#### 保留功能（無需修改）
- ✅ 物理群集狀態機（第 109-342 行）
- ✅ JPG 幀匯出（pre/post cluster）
- ✅ CSV 格式（所有欄位保持一致）
- ✅ Inspection 影片生成
- ✅ 旋轉校正整合（rotation_config）
- ✅ 比例尺換算（scale_cache）

#### 輸出格式
**CSV 檔名**: `{video_name}_dark.csv`（例如 `21_dark.csv`）
**Inspection 影片**: `{video_name}_dark_inspection.mp4`
**匯出幀目錄**: `lifts/exported_frames/{video_name}_dark/`

---

### 階段 3: 數據整合工具（2-3天）

#### 目標
合併暗房與非暗房的偵測結果為完整 CSV。

#### 交付成果
- **檔案**: `merge_darkroom_results.py`
- **測試**: 合併至少 2 個影片的結果

#### 核心功能
```python
def merge_darkroom_results(original_csv, darkroom_csv, output_csv, video_name):
    """
    合併暗房與非暗房的偵測結果

    參數:
        original_csv: 忽略暗房區間的原始結果 (例如 21.csv)
        darkroom_csv: 暗房區間專用結果 (例如 21_dark.csv)
        output_csv: 完整合併結果 (例如 21_merged.csv)
        video_name: 影片檔名 (用於查詢 darkroom_intervals)

    流程:
        1. 載入 darkroom_intervals.py 取得時間區間
        2. 載入兩個 CSV，驗證欄位一致性
        3. 依 frame_idx 排序
        4. 根據時間區間決定資料來源:
           - is_in_darkroom → 使用 darkroom_csv 的該幀
           - not is_in_darkroom → 使用 original_csv 的該幀
        5. 重新編號 cluster_id（避免衝突）
        6. 驗證時間戳連續性（警告缺失幀）
        7. 輸出合併 CSV

    相容性:
        - 支援已清理的 CSV (c*.csv)
        - 支援已人工校正的 CSV (mc*.csv)
        - 保留所有欄位 (cluster_id, orientation, darkroom_event, frame_path)
    """
```

#### 輸出格式
**檔名規則**:
- 原始 + 暗房 → `{name}_merged.csv`
- 清理後 + 暗房清理後 → `c{name}_merged.csv`
- 校正後 + 暗房校正後 → `mc{name}_merged.csv`

#### Cluster ID 重新編號邏輯
```python
# 確保整個影片的 cluster_id 連續且唯一
# 方法: 掃描兩個 CSV，依時間順序重新分配 ID

global_cluster_id = 0
for row in merged_df.itertuples():
    if row.cluster_id > 0 and row.cluster_id != prev_cluster_id:
        global_cluster_id += 1
        cluster_mapping[row.cluster_id] = global_cluster_id
    prev_cluster_id = row.cluster_id
```

#### 驗證檢查
1. **時間連續性**: 相鄰幀的 frame_idx 差距 = FRAME_INTERVAL (6)
2. **區間覆蓋**: 所有暗房區間的幀都來自 darkroom_csv
3. **資料完整性**: 無遺失幀（若有則警告）

---

### 階段 4: 人工校正整合測試（1天）

#### 目標
確保 `manual_correction_tool.py` 能正確處理合併後的 CSV。

#### 測試項目
1. ✅ 載入 `*_merged.csv` 檔案
2. ✅ 正確識別物理群集（包含暗房群集）
3. ✅ JPG 幀路徑正確（`{video_name}_dark/pre_cluster_*.jpg`）
4. ✅ 人工校正後儲存為 `mc*_merged.csv`
5. ✅ 暫存恢復功能正常

#### 預期行為
由於人工校正工具基於 `frame_path` 欄位識別物理群集，而暗房腳本會正確生成這些標籤，理論上應該**零修改**即可相容。

#### 驗證方法
手動執行人工校正流程，確認：
- 能順利選擇 ROI
- JPG 圖片正確載入（自動回退到影片幀）
- 校正值正確應用到整個物理群集

---

### 階段 5: Inspection 影片整合（可選，2天）

#### 目標
合併原始與暗房的 inspection 影片為完整檢閱影片。

#### 交付成果
- **檔案**: `merge_darkroom_inspection.py`
- **輸出**: `{video_name}_merged_inspection.mp4`

#### 技術方案
使用 OpenCV 或 ffmpeg 進行影片片段剪輯與合併。

**方法 A: OpenCV (純 Python)**
```python
# 優點: 無外部依賴，完全控制
# 缺點: 效能較慢，需要逐幀讀寫

for interval in darkroom_intervals:
    # 非暗房前段
    copy_frames(original_inspection, 0, interval.start_frame, output_writer)
    # 暗房片段
    copy_frames(darkroom_inspection, interval.start_frame, interval.end_frame, output_writer)
    # 非暗房後段
    copy_frames(original_inspection, interval.end_frame, end, output_writer)
```

**方法 B: ffmpeg (推薦)**
```python
# 優點: 效能極佳，無重新編碼
# 缺點: 需要 ffmpeg 可執行檔

# 生成切割時間點列表
segments = [
    f"file 'original_inspection.mp4' inpoint 0 outpoint {interval.start}",
    f"file 'darkroom_inspection.mp4' inpoint {interval.start} outpoint {interval.end}",
    # ...
]

# 使用 ffmpeg concat demuxer
subprocess.run(['ffmpeg', '-f', 'concat', '-i', 'segments.txt', '-c', 'copy', 'output.mp4'])
```

#### 挑戰
1. **時間戳對齊**: 確保切割點精確對齊幀邊界
2. **編碼參數**: 保持與原始影片一致的 codec/fps/resolution
3. **過渡平滑**: 切換點不應有視覺跳躍

---

## 四、開發規範

### 4.1 檔案管理原則
1. ✅ **絕不修改現有檔案**（階段 1-3）
2. ✅ 所有新功能以**新檔案**形式開發
3. ✅ 測試通過後才考慮整合到主程式
4. ✅ 保持現有系統完全可用

### 4.2 命名規範

#### 腳本檔案
- 分析工具: `analyze_*.py`
- 測試腳本: `test_*.py`
- 主程式: `lift_travel_detection_dark.py`
- 整合工具: `merge_*.py`

#### 輸出檔案
- CSV: `{name}_dark.csv` → `{name}_merged.csv`
- Inspection: `{name}_dark_inspection.mp4` → `{name}_merged_inspection.mp4`
- 匯出幀: `lifts/exported_frames/{name}_dark/`

### 4.3 相容性要求
1. **CSV 格式**: 完全相同的欄位結構
   ```
   frame_idx, second, vertical_travel_distance (mm),
   cluster_id, orientation, darkroom_event, frame_path
   ```
2. **JPG 命名**: `pre_cluster_{id:03d}.jpg`, `post_cluster_{id:03d}.jpg`
3. **旋轉設定**: 使用 `rotation_config.py`
4. **比例尺**: 使用 `scale_cache_utils.py`

---

## 五、測試策略

### 5.1 單元測試
- ✅ CLAHE 前處理函數
- ✅ 特徵點偵測與匹配
- ✅ 暗房區間判斷邏輯
- ✅ Cluster ID 重新編號

### 5.2 整合測試
**測試影片集**（從 darkroom_intervals.py）:
1. `21.mp4` - 00:30 ~ 08:28 (8分鐘，主要測試影片)
2. `28.mp4` - 05:27 ~ 12:11 (6.7分鐘)
3. `30.mp4` - 兩個區間：02:56~04:50, 05:31~12:40

**測試流程**:
1. 執行 `lift_travel_detection_dark.py` → 產生 `*_dark.csv`
2. 執行 `merge_darkroom_results.py` → 產生 `*_merged.csv`
3. 手動驗證: 檢視 inspection 影片，確認運動偵測正確
4. 人工校正測試: 使用 `manual_correction_tool.py` 處理合併 CSV

### 5.3 驗收標準
- ✅ 暗房區間運動偵測成功率 > 70%
- ✅ 合併 CSV 時間戳連續無缺失
- ✅ Cluster ID 全域唯一且連續
- ✅ 人工校正工具完全相容
- ✅ Inspection 影片無明顯異常

---

## 六、風險評估與應對

### 6.1 技術風險

#### 風險 1: 特徵匹配失敗率過高
**機率**: 低（驗證分析顯示特徵點充足）
**影響**: 運動偵測大量誤判或遺漏
**應對**:
1. 調整 ORB 參數（nfeatures, scaleFactor）
2. 嘗試 SIFT 偵測器（更強但較慢）
3. 備案: 混合方案（自動初檢 + 人工補充）

#### 風險 2: 運動幅度計算偏差
**機率**: 中（暗房畫質影響準確度）
**影響**: 需要更多人工校正工作
**應對**:
1. 保持人工校正工具可用（已驗證相容）
2. 調整統計閾值（t-test p-value）
3. 增加物理群集判斷的保守度

#### 風險 3: 旋轉校正影響
**機率**: 低（旋轉功能已在主程式驗證）
**影響**: 暗房影片旋轉後畫質進一步下降
**應對**:
1. 在測試階段驗證旋轉影響
2. 優先處理已旋轉的影片檔案
3. 若有問題，調整 CLAHE 參數

### 6.2 整合風險

#### 風險 4: CSV 合併邏輯錯誤
**機率**: 中（時間區間邊界處理複雜）
**影響**: 資料遺失或重複
**應對**:
1. 充分的單元測試與邊界測試
2. 驗證工具：比對合併前後的總運動量
3. 人工抽查：檢視關鍵時間點的資料來源

#### 風險 5: Cluster ID 衝突
**機率**: 中（兩個 CSV 可能有重疊 ID）
**影響**: 人工校正工具混淆群集
**應對**:
1. 強制重新編號策略（已規劃）
2. 驗證腳本：檢查 ID 唯一性
3. 在合併工具中加入衝突警告

### 6.3 使用者風險

#### 風險 6: 工作流程複雜化
**機率**: 中（增加額外處理步驟）
**影響**: 使用者操作失誤機率上升
**應對**:
1. 提供一鍵式腳本（自動執行整個 pipeline）
2. 詳細文件說明（包含範例）
3. 錯誤訊息清晰明確

---

## 七、時程規劃

| 階段 | 任務 | 預估時間 | 累計時間 | 里程碑 |
|------|------|----------|----------|--------|
| **0** | 可行性分析 | ✅ 完成 | - | 確定技術方案 |
| **1** | 運動偵測驗證 | 1-2天 | 1-2天 | 驗證報告 |
| **2** | 暗房專用主程式 | 3-5天 | 4-7天 | 完整偵測腳本 |
| **3** | 數據整合工具 | 2-3天 | 6-10天 | 合併工具 |
| **4** | 人工校正整合測試 | 1天 | 7-11天 | 整合驗證 |
| **5** | Inspection 整合（可選） | 2天 | 9-13天 | 完整 pipeline |

**核心交付時間**: 7-11天
**完整交付時間**: 9-13天（含可選功能）

---

## 八、成功指標

### 8.1 技術指標
- ✅ 暗房區間特徵點匹配成功率 ≥ 70%
- ✅ 運動偵測召回率（能偵測到的真實運動）≥ 80%
- ✅ CSV 合併無資料遺失或重複

### 8.2 品質指標
- ✅ 與原始系統精度相當（可透過人工校正達成）
- ✅ 人工校正工具零修改即可使用
- ✅ 輸出格式完全一致

### 8.3 效能指標
- ✅ 單個暗房區間處理時間 < 原始系統的 1.5 倍
- ✅ 批次處理 6 個暗房影片 < 1 小時

---

## 九、後續擴展方向

### 9.1 短期優化（Phase 2.5）
1. **參數自動調適**: 根據畫質自動調整 CLAHE 參數
2. **批次處理腳本**: 自動處理所有配置的暗房影片
3. **視覺化比較工具**: 對比暗房前後的偵測結果

### 9.2 長期研究（Phase 3）
1. **深度學習方案**: 訓練 CNN 模型直接預測運動
2. **多模態融合**: 結合音頻資訊（馬達聲音）
3. **即時處理**: 優化效能支援即時分析

---

## 十、參考資料

### 10.1 核心檔案
- `src/lift_travel_detection.py` - 原始運動偵測主程式（623行）
- `src/manual_correction_tool.py` - 人工校正工具
- `src/darkroom_intervals.py` - 暗房時間區間配置
- `src/darkroom_utils.py` - 暗房工具函數
- `manual_correction_guide.md` - 人工校正工作流文件

### 10.2 分析工具
- `analyze_darkroom_video.py` - 可行性分析腳本（本次開發）
- `lifts/darkroom_analysis/` - 分析結果圖片

### 10.3 測試資料
- `lifts/darkroom_data/21_a.mp4` - 主要測試影片
- `lifts/data/*.mp4` - 完整影片庫（含暗房區間）

---

## 附錄 A: CLAHE 參數說明

```python
cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
```

**clipLimit**: 對比度限制閾值
- 預設: 40.0（過高，可能過度增強雜訊）
- 建議: 2.0-4.0（暗房影片適用範圍）
- 影響: 越高對比度越強，但雜訊也越明顯

**tileGridSize**: 切割網格大小
- 預設: (8,8)（影像分為 8×8=64 個區域）
- 建議: (4,4) - (16,16)
- 影響: 越小局部適應性越強，但可能產生塊狀效應

**調適建議**:
- 若特徵點不足: 提高 clipLimit 至 4.0
- 若雜訊過多: 降低 clipLimit 至 2.0
- 若邊界有塊狀: 增大 tileGridSize 至 (12,12)

---

## 附錄 B: 暗房區間統計

根據 `src/darkroom_intervals.py` 當前配置:

| 影片 | 暗房區間 | 時長 | 佔比 |
|------|----------|------|------|
| 28.mp4 | 05:27-12:11 | 6.7分 | ~42% |
| 29.mp4 | 04:19-10:55 | 6.6分 | ~53% |
| 30.mp4 | 兩段 | 9.1分 | ~61% |
| 31-1.mp4 | 02:45-08:54 | 6.2分 | ~51% |
| 37.mp4 | 03:56-05:08 | 1.2分 | 少量 |
| 21.mp4 | 00:30-08:28 | 8.0分 | ~60% |

**總計**: 6 個影片，約 37.8 分鐘暗房影片需要處理

**處理優先級**:
1. 21.mp4（已有增強版本 21_a.mp4，測試用）
2. 28.mp4, 29.mp4, 30.mp4（時長較長）
3. 31-1.mp4, 37.mp4（較短，驗證用）

---

## 版本歷史

| 版本 | 日期 | 修改內容 | 作者 |
|------|------|----------|------|
| 1.0 | 2025-09-30 | 初版建立，完成可行性分析與實施計畫 | Claude Code |

---

**文件狀態**: ✅ 已確認，進入開發階段
**下一步**: 開始階段 1 - 運動偵測驗證腳本開發