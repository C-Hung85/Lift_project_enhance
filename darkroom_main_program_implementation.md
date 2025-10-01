# 暗房主程式實施計畫

**日期**: 2025-09-30
**狀態**: 📋 計畫階段
**目標**: 建立專用於暗房區間運動偵測的主程式

---

## 一、專案背景

### Phase 1 測試結論

經過短片段測試 (`21a_tinyclip.mp4`)，確認以下結論：

1. ✅ **CLAHE 前處理有效**
   - `clipLimit=3.0, tileGridSize=(8,8)`
   - 在暗房環境下成功偵測到連續微小運動 (-2px/幀)
   - 特徵點集中在尺標上（正確的追蹤目標）

2. ✅ **運動特性確認**
   - 暗房區間運動緩慢（每幀約 -2px）
   - 向上移動為負值
   - 需要極低的位移門檻（~1.5px）

3. ✅ **Canny Edge Detection 不適用**
   - 雖能捕捉顯微鏡邊緣，但靜態邊緣為主
   - 邊緣閃爍導致匹配錯誤（紅線綠線亂飛）
   - 產生誤報（-40px）

4. ✅ **主程式設計模式正確**
   - 順序讀取（避免 OpenCV frame navigation 問題）
   - T-test 統計檢定（而非固定門檻）
   - 物理群集狀態機（防止正負抖動對）

---

## 二、實施目標

### 核心目標

建立 `lift_travel_detection_dark.py`，專門處理暗房區間運動偵測，並產生與主程式完全對應的輸出格式，便於後續資料融合。

### 設計原則

1. **完整繼承主程式架構**
   - 保留順序讀取邏輯
   - 保留 T-test 統計檢定
   - 保留物理群集狀態機
   - 保留 inspection video 與 JPG 匯出功能

2. **反轉暗房邏輯**
   - 原主程式：非暗房處理，暗房填 0
   - 新暗房程式：暗房處理，非暗房填 0

3. **輸出格式一致性**
   - CSV 長度與主程式完全一致
   - Inspection video 長度一致
   - 幀索引對應一致
   - 便於後續融合

---

## 三、技術規格

### 3.1 檔案與路徑處理

#### 輸入檔案位置
```python
# 原主程式
INPUT_DIR = os.path.join(LIFT_BASE, 'data')  # lifts/data/

# 新暗房程式
INPUT_DIR = os.path.join(LIFT_BASE, 'darkroom_data')  # lifts/darkroom_data/
```

#### 檔案命名規則
```
輸入檔案: 21a.mp4 (darkroom_data/)
參數檔案: 21.mp4 (config.py, rotation_config.py, scale_images/)

命名轉換邏輯:
- 處理檔案: 21a.mp4
- 查詢 rotation: '21.mp4' (去除 _a)
- 查詢 scale: '21.mp4' (去除 _a)
- 查詢 config: '21.mp4' (去除 _a)
```

#### 參數查詢函數
```python
def get_base_video_name(filename):
    """
    將 darkroom 檔名轉換為 base 檔名以查詢參數

    Examples:
        '21a.mp4' -> '21.mp4'
        '21_a.mp4' -> '21.mp4'
        '21.mp4' -> '21.mp4'
    """
    name, ext = os.path.splitext(filename)
    # 移除 _a 或 a 後綴
    if name.endswith('_a'):
        name = name[:-2]
    elif name.endswith('a'):
        name = name[:-1]
    return name + ext
```

### 3.2 暗房邏輯反轉

#### 原主程式邏輯（Line 201-238）
```python
# 在暗房區間內，將運動距離設為 0（忽略）
if is_darkroom:
    vertical_travel_distance = 0

# 暗房事件
if is_darkroom and not prev_is_darkroom:
    darkroom_event = 'enter_darkroom'
elif (not is_darkroom) and prev_is_darkroom:
    darkroom_event = 'exit_darkroom'

# 候選判定
is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and (not is_darkroom)

# 進入暗房時強制結束群集
if is_darkroom:
    if state == 'InCluster':
        # 匯出 post frame
        # 重置狀態機
    state = 'Idle'
```

#### 新暗房程式邏輯（反轉）
```python
# 在非暗房區間內，將運動距離設為 0（忽略）
if not is_darkroom:
    vertical_travel_distance = 0

# 暗房事件（名稱保持一致）
if is_darkroom and not prev_is_darkroom:
    darkroom_event = 'enter_darkroom'
elif (not is_darkroom) and prev_is_darkroom:
    darkroom_event = 'exit_darkroom'

# 候選判定（反轉：只處理暗房）
is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and is_darkroom

# 離開暗房時強制結束群集
if not is_darkroom:
    if state == 'InCluster':
        # 匯出 post frame
        # 重置狀態機
    state = 'Idle'
```

### 3.3 CLAHE 前處理整合

#### 整合位置
在特徵偵測前加入 CLAHE 前處理：

```python
# 原主程式 (Line 126)
keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame, mask)

# 新暗房程式（加入 CLAHE）
def preprocess_darkroom_frame(frame):
    """CLAHE 前處理（僅用於暗房區間）"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return enhanced

# 在偵測前套用（只在暗房區間）
if is_darkroom:
    frame_for_detection = preprocess_darkroom_frame(frame)
else:
    frame_for_detection = frame

keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame_for_detection, mask)
```

#### 注意事項
- CLAHE 只在暗房區間套用（非暗房區間不處理）
- 確保 inspection video 顯示的是 CLAHE 增強後的效果
- 保留原始 frame 用於 JPG 匯出（或匯出 CLAHE 版本？待決定）

### 3.4 輸出規格

#### CSV 輸出
```python
# 原主程式
OUTPUT_PATH = os.path.join(LIFT_BASE, 'result', f'{video_name}.csv')

# 新暗房程式
OUTPUT_PATH = os.path.join(LIFT_BASE, 'result', f'{video_name}_dark.csv')
# 例如: lifts/result/21a_dark.csv
```

#### Inspection Video
```python
# 原主程式
INSPECTION_PATH = os.path.join(LIFT_BASE, 'inspection', f'{video_name}_inspection.mp4')

# 新暗房程式
INSPECTION_PATH = os.path.join(LIFT_BASE, 'inspection', f'{video_name}_dark_inspection.mp4')
# 例如: lifts/inspection/21a_dark_inspection.mp4
```

#### JPG 匯出
```python
# 原主程式
JPG_DIR = os.path.join(LIFT_BASE, 'exported_frames', video_name)

# 新暗房程式
JPG_DIR = os.path.join(LIFT_BASE, 'exported_frames', f'{video_name}_dark')
# 例如: lifts/exported_frames/21a_dark/pre_cluster_001.jpg
```

#### CSV 格式保持一致
```csv
frame,frame_idx,keypoints,camera_pan,v_travel_distance,kp_pair_lines,frame_path,cluster_id,orientation,darkroom_event
0.0,1800,22,False,0.0,"[[...]]",,0,0,
0.1,1806,23,False,0.0,"[[...]]",,0,0,enter_darkroom
0.2,1812,24,False,12.5,"[[...]]",pre_cluster_001.jpg,1,1,
...
```

---

## 四、修改項目清單

### 4.1 檔案與路徑
- [ ] 複製 `src/lift_travel_detection.py` → `src/lift_travel_detection_dark.py`
- [ ] 修改 `INPUT_DIR` 為 `darkroom_data`
- [ ] 新增 `get_base_video_name()` 函數
- [ ] 修改所有參數查詢使用 base name

### 4.2 暗房邏輯反轉
- [ ] 反轉候選判定條件：`and is_darkroom`
- [ ] 反轉強制結束條件：`if not is_darkroom`
- [ ] 反轉位移歸零條件：`if not is_darkroom`
- [ ] 保持 darkroom_event 名稱不變

### 4.3 CLAHE 前處理
- [ ] 新增 `preprocess_darkroom_frame()` 函數
- [ ] 在特徵偵測前條件套用 CLAHE
- [ ] 建立 `frame_enhanced_bgr` 變數（暗房用 CLAHE，非暗房用原始）
- [ ] 替換所有使用 `frame` 的地方為 `frame_enhanced_bgr`（視覺化、JPG、inspection video）
- [ ] 確保 inspection video 顯示 CLAHE 增強效果
- [ ] 確保 JPG 匯出使用 CLAHE 增強版本

### 4.4 輸出檔名
- [ ] CSV: 加上 `_dark` 後綴
- [ ] Inspection video: 加上 `_dark` 後綴（目錄沿用 `lifts/inspection`）
- [ ] JPG 目錄: 加上 `_dark` 後綴

### 4.5 運動方向一致性
- [ ] 確認垂直位移計算保持不變（向上 = 正值）
- [ ] 檢查 orientation 定義（上升 = +1，下降 = -1）

### 4.6 測試驗證
- [ ] 使用 `21a.mp4` 測試完整 pipeline
- [ ] 驗證 CSV 長度與原主程式一致
- [ ] 驗證暗房區間有運動資料，非暗房區間全為 0
- [ ] 檢查 inspection video 與 JPG 匯出

---

## 五、程式碼修改範例

### 5.1 檔案路徑處理

```python
# ===== 新增：Base name 轉換函數 =====
def get_base_video_name(filename):
    """
    將 darkroom 檔名轉換為 base 檔名以查詢參數

    Examples:
        '21a.mp4' -> '21.mp4'
        '21_a.mp4' -> '21.mp4'
        '21.mp4' -> '21.mp4'
    """
    name, ext = os.path.splitext(filename)
    # 移除 _a 或 a 後綴
    if name.endswith('_a'):
        name = name[:-2]
    elif name.endswith('a'):
        name = name[:-1]
    return name + ext

# ===== 修改：輸入目錄 =====
# 原: INPUT_DIR = os.path.join(LIFT_BASE, 'data')
INPUT_DIR = os.path.join(LIFT_BASE, 'darkroom_data')

# ===== 修改：參數查詢 =====
file_name = os.path.basename(file_path)
base_name = get_base_video_name(file_name)  # 新增

# 查詢 rotation
rotation_angle = rotation_dict.get(base_name, 0.0)  # 使用 base_name

# 查詢 scale
scale_factor = video_scale_dict.get(base_name, 1.0)  # 使用 base_name

# 查詢 config
video_config_entry = video_config.get(base_name, {})  # 使用 base_name

# 查詢 darkroom intervals
darkroom_intervals = DARKROOM_INTERVALS.get(base_name, [])  # 使用 base_name

# ===== 修改：輸出檔名 =====
video_name = os.path.splitext(file_name)[0]  # 保持為 '21a'（不去 _a）

# CSV
output_file = os.path.join(LIFT_BASE, 'result', f'{video_name}_dark.csv')

# Inspection video
inspection_output = os.path.join(LIFT_BASE, 'inspection', f'{video_name}_dark_inspection.mp4')

# JPG 目錄
jpg_dir = os.path.join(LIFT_BASE, 'exported_frames', f'{video_name}_dark')
```

### 5.2 暗房邏輯反轉

```python
# ===== 修改：位移歸零條件（反轉）=====
# 原: if is_darkroom:
#         vertical_travel_distance = 0
if not is_darkroom:
    vertical_travel_distance = 0

# ===== 修改：候選判定（反轉）=====
# 原: is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and (not is_darkroom)
is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and is_darkroom

# ===== 修改：強制結束群集（反轉）=====
# 原: if is_darkroom:
if not is_darkroom:
    # 非暗房：強制回到 Idle 狀態，不累計群集
    if state == 'InCluster':
        # 在群內時離開暗房 → 強制出群，post 用離開前最後一個暗房幀
        if last_darkroom_frame is not None and current_cluster_id:
            post_name = f'post_cluster_{current_cluster_id:03d}.jpg'
            export_frame_jpg(last_darkroom_frame, post_name, video_name)
            # 將上一幀的 frame_path 標記為 post
            if 'frame_path' in result and len(result['frame_path']) > 0:
                result['frame_path'][-1] = post_name
        # 匯出 pre（若尚未匯出）
        if pending_pre_export is not None:
            export_frame_jpg(pending_pre_export[0], pending_pre_export[1], video_name)
            pending_pre_export = None
    state = 'Idle'
    pending_idx = None
    pending_delta_px = None
    pending_result_idx = None
    zero_streak = 0
    reversal_streak = 0
    orientation_current = 0
    current_cluster_id = 0

# ===== 修改：暗房幀快取（名稱改為 last_darkroom_frame）=====
# 原: last_non_darkroom_frame = (frame_idx, frame) if not is_darkroom else last_non_darkroom_frame
last_darkroom_frame = (frame_idx, frame_enhanced_bgr) if is_darkroom else last_darkroom_frame
```

### 5.3 CLAHE 前處理整合

```python
# ===== 新增：CLAHE 前處理函數 =====
def preprocess_darkroom_frame(frame):
    """
    CLAHE 前處理（僅用於暗房區間）

    Parameters:
        frame: BGR or grayscale frame

    Returns:
        enhanced: Grayscale enhanced frame
    """
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return enhanced

# ===== 修改：特徵偵測前套用 CLAHE =====
# 在主迴圈中，特徵偵測前加入條件處理

# 判斷是否在暗房區間
is_darkroom = is_in_darkroom_interval(frame_idx / fps, darkroom_intervals)

# 準備用於偵測和視覺化的幀
if is_darkroom:
    # CLAHE 前處理
    frame_gray_enhanced = preprocess_darkroom_frame(frame)
    # 轉換為 BGR 用於視覺化和 JPG 匯出
    frame_enhanced_bgr = cv2.cvtColor(frame_gray_enhanced, cv2.COLOR_GRAY2BGR)
    # 用於特徵偵測
    frame_for_detection = frame_gray_enhanced
else:
    # 非暗房區間使用原始幀
    frame_enhanced_bgr = frame
    frame_for_detection = frame

# 偵測特徵
keypoint_list2, feature_descrpitor2 = feature_detector.detectAndCompute(frame_for_detection, mask)

# 重要：後續所有視覺化、JPG 匯出、inspection video 都使用 frame_enhanced_bgr
# 這樣可以確保：
# 1. 暗房區間：使用 CLAHE 增強版本
# 2. 非暗房區間：使用原始版本
# 3. 兩者無縫銜接
```

**關鍵修改點**：
1. **所有使用 `frame` 的地方改為 `frame_enhanced_bgr`**：
   - `draw_keypoints()` 函數參數
   - `export_frame_jpg()` 函數參數
   - Inspection video 寫入
   - Frame cache 快取

2. **確保方向一致性**：
   - 原主程式：向上 = 正值
   - 保持一致，無需修改垂直位移計算
```

---

## 六、測試計畫

### 6.1 初步測試

**測試檔案**: `21a.mp4`

**驗證項目**:
1. 程式能正確載入 `darkroom_data/21a.mp4`
2. 正確查詢 `21.mp4` 的 rotation (21°), scale, config
3. 正確識別暗房區間（00:30 - 08:28）

**預期輸出**:
- `lifts/result/21a_dark.csv`
- `lifts/inspection/21a_dark_inspection.mp4`
- `lifts/exported_frames/21a_dark/pre_cluster_XXX.jpg`

### 6.2 輸出驗證

#### CSV 驗證
```python
import pandas as pd

# 載入兩份 CSV
df_main = pd.read_csv('lifts/result/21.csv')
df_dark = pd.read_csv('lifts/result/21a_dark.csv')

# 驗證長度一致
assert len(df_main) == len(df_dark), "CSV 長度不一致"

# 驗證幀索引對應
assert (df_main['frame_idx'] == df_dark['frame_idx']).all(), "幀索引不對應"

# 驗證互補性
# 主程式：非暗房有資料，暗房為 0
# 暗房程式：暗房有資料，非暗房為 0
```

#### Inspection Video 驗證
- 播放檢查暗房區間是否顯示 CLAHE 增強效果
- 檢查非暗房區間是否無特徵點標記
- 檢查群集 JPG 是否正確匯出

### 6.3 功能驗證

- [ ] CLAHE 前處理在暗房區間生效
- [ ] 特徵點集中在尺標區域
- [ ] 偵測到微小運動（約 -2px/幀）
- [ ] 物理群集正確形成
- [ ] 進出暗房時群集正確結束
- [ ] T-test 統計檢定正常運作

---

## 七、後續融合策略

### 資料融合方法

```python
import pandas as pd

# 載入兩份結果
df_main = pd.read_csv('lifts/result/21.csv')        # 非暗房區間
df_dark = pd.read_csv('lifts/result/21a_dark.csv')  # 暗房區間

# 融合邏輯：取非零值
df_merged = df_main.copy()

# 運動距離：取非零值
mask = df_dark['v_travel_distance'] != 0
df_merged.loc[mask, 'v_travel_distance'] = df_dark.loc[mask, 'v_travel_distance']

# 群集 ID：取非零值（需重新編號避免衝突）
mask = df_dark['cluster_id'] != 0
max_cluster_id = df_main['cluster_id'].max()
df_merged.loc[mask, 'cluster_id'] = df_dark.loc[mask, 'cluster_id'] + max_cluster_id

# 方向：取非零值
mask = df_dark['orientation'] != 0
df_merged.loc[mask, 'orientation'] = df_dark.loc[mask, 'orientation']

# Frame path：取非空值
mask = df_dark['frame_path'] != ''
df_merged.loc[mask, 'frame_path'] = df_dark.loc[mask, 'frame_path']

# 儲存融合結果
df_merged.to_csv('lifts/result/21_merged.csv', index=False)
```

### Inspection Video 融合

使用 FFmpeg 拼接：
```bash
ffmpeg -i 21_inspection.mp4 -i 21a_dark_inspection.mp4 \
       -filter_complex "[0:v][1:v]blend=all_expr='if(eq(A,0),B,A)'" \
       21_merged_inspection.mp4
```

或使用 Python 逐幀融合（更精確控制）。

---

## 八、風險與注意事項

### 8.1 已知風險

1. **CLAHE 在非暗房區間的副作用**
   - 解決方案：只在暗房區間套用 CLAHE

2. **Base name 轉換邏輯錯誤**
   - 風險：找不到 rotation/scale 參數
   - 解決方案：詳細測試 `get_base_video_name()` 函數

3. **CSV 長度不一致**
   - 風險：融合時無法對齊
   - 解決方案：使用相同的起訖點配置

4. **群集 ID 衝突**
   - 風險：融合時兩份 CSV 的 cluster_id 重複
   - 解決方案：融合時重新編號

### 8.2 已確認決策

1. **JPG 匯出版本** ✅
   - **決定**: 匯出 CLAHE 增強版本
   - 理由：便於人工校正時觀察細節

2. **Inspection Video CLAHE 顯示** ✅
   - **決定**: Inspection video 使用 CLAHE 處理後的畫面
   - 實作：修改視覺化代碼使用 `frame_enhanced_bgr`

3. **運動方向定義** ✅
   - **決定**: 向上移動 = 正值（與原主程式一致）
   - **注意**: 測試腳本中是負值，需要在暗房主程式中保持與原主程式一致

4. **參數設定** ✅
   - `EFFECT_MIN_PX = 1.0` (保持一致)
   - `FRAME_INTERVAL = 6` (保持一致)
   - 其他物理群集參數保持一致

5. **Inspection 輸出目錄** ✅
   - **決定**: 沿用 `D:\Lift_project\lifts\inspection`
   - 檔名已有差異 (`21a_dark_inspection.mp4`)，無衝突問題

---

## 九、實施時程

| 階段 | 預估時間 | 任務 |
|------|----------|------|
| **階段 1** | 30 分鐘 | 複製檔案、修改路徑與檔名邏輯 |
| **階段 2** | 30 分鐘 | 反轉暗房邏輯、修改條件判斷 |
| **階段 3** | 20 分鐘 | 整合 CLAHE 前處理 |
| **階段 4** | 30 分鐘 | 初步測試、除錯 |
| **階段 5** | 20 分鐘 | 輸出驗證、功能確認 |
| **總計** | **2-3 小時** | |

---

## 十、成功標準

### 必須達成
- [x] ✅ 程式能正確處理 `darkroom_data/*.mp4` 檔案
- [ ] ✅ 正確查詢 base name 的參數（rotation, scale, config）
- [ ] ✅ CSV 長度與原主程式完全一致
- [ ] ✅ 暗房區間有運動資料，非暗房區間全為 0
- [ ] ✅ CLAHE 在暗房區間生效
- [ ] ✅ 物理群集邏輯正常運作
- [ ] ✅ 進出暗房時群集正確結束

### 期望達成
- [ ] ✅ 偵測到測試短片中的微小運動（-2px/幀）
- [ ] ✅ 特徵點集中在尺標區域
- [ ] ✅ Inspection video 清晰顯示 CLAHE 效果
- [ ] ✅ JPG 正確匯出（pre/post cluster）

---

## 十一、相關檔案

### 輸入檔案
- `lifts/darkroom_data/21a.mp4` - 暗房測試影片
- `src/config.py` - 起訖點配置（查詢 21.mp4）
- `src/rotation_config.py` - 旋轉角度（查詢 21.mp4）
- `src/scale_config.py` - 尺度因子（查詢 21.mp4）
- `src/darkroom_intervals.py` - 暗房時間區間（查詢 21.mp4）

### 程式檔案
- `src/lift_travel_detection.py` - 原主程式（參考）
- `src/lift_travel_detection_dark.py` - 新暗房主程式（待建立）
- `src/darkroom_utils.py` - 暗房工具函數
- `src/rotation_utils.py` - 旋轉工具函數

### 測試檔案
- `test_tinyclip_v3.py` - 短片段測試腳本（CLAHE 版本）
- `lifts/test_short/21a_tinyclip.mp4` - 測試短片

### 輸出檔案（預期）
- `lifts/result/21a_dark.csv` - 暗房運動分析結果
- `lifts/inspection/21a_dark_inspection.mp4` - 暗房 inspection video
- `lifts/exported_frames/21a_dark/` - 暗房群集 JPG

---

**文件版本**: 1.0
**最後更新**: 2025-09-30
**作者**: Claude Code
**狀態**: ✅ 計畫完成，等待確認細節