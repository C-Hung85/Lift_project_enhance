# 半自動人工校正工作流開發指引

## 概要

這個工作流程用於手動校正電梯位移檢測數據，通過高精度的視覺化線段標記讓使用者提供真實的位移參考，校正自動檢測結果。支援多檔案格式、智能暫存恢復、以及物理群集精確定位。

## 系統架構

### 核心模組
- **影片處理**: `lifts/data/*.mp4` - 原始影片檔案
- **旋轉校正**: `src/rotation_config.py` + `src/rotation_utils.py` - 影片旋轉處理
- **比例尺換算**: `src/scale_config.py` - 像素到毫米的換算係數
- **分析數據**: `lifts/result/*.csv` - 主程式輸出的位移數據（支援多種格式）
- **預匯出幀**: `lifts/exported_frames/{video_name}/` - 物理群集參考幀
- **暫存系統**: `lifts/result/*_temp_*.json` - 工作狀態暫存檔案

### 工作流程圖
```
原始影片 → 旋轉校正 → 物理群集檢測 → JPG幀匯出 → 智能CSV載入 → 暫存檢測 → ROI選擇 → 8倍放大標記 → 3次平均校正 → 智能儲存
    ↓           ↓           ↓            ↓           ↓           ↓         ↓         ↓            ↓            ↓
  data/     rotation    motion       exported     flexible    temp      user      8x zoom     avg+outlier   temp/final
 *.mp4      utils     clustering      JPG        format      resume    ROI      precision    removal       output
```

## 詳細工作流程

### 階段 0: 智能初始化
1. **智能檔案載入**: 支援多種CSV格式 (`1.csv`, `c1.csv`, `mc1.csv`)
2. **暫存檢測**: 自動搜尋同名暫存檔案 (`{name}_temp_{timestamp}.json`)
3. **暫存選擇**: 多個暫存檔案時提供選擇介面，顯示建立時間和進度
4. **狀態恢復**: 載入暫存狀態，包括進度、標註記錄、CSV修改
5. **欄位智能檢測**: 
   - **位移欄位**: 自動識別順序 → `displacement` / `displacement_mm` / `位移` / `位移_mm` → 第3欄（按位置）
   - **時間軸欄位**: 檢測 `second` 欄位（時戳，必需）和 `frame_idx` 欄位（幀號，用於顯示）
   - **群集標籤**: 檢測 `frame_path` 欄位（物理群集標籤系統必需，缺少則報錯）
6. **載入配置**: 比例尺 (`scale_config`)、旋轉角度 (`rotation_config`)、物理群集標籤
7. **預匯出幀檢測**: 檢查 `lifts/exported_frames/{video_name}/` 目錄中的JPG參考幀

### 階段 1: JPG檔案加載（必需）
- **JPG檔案必需**: 工具要求必須擁有所有物理群集的JPG參考幀
  - `frame_path` 格式: `pre_cluster_XXX.jpg` 或 `post_cluster_XXX.jpg`（XXX為群集序號)
  - 自動識別前零點（`pre_cluster_*`）和後零點（`post_cluster_*`）
  - 缺少JPG檔案會報錯並停止執行
- **JPG檔案路徑**: `lifts/exported_frames/{video_name}/pre_cluster_XXX.jpg`
- **旋轉校正整合**: JPG加載時自動應用 `rotation_config` 設定的旋轉角度
- **精確定位**: 使用預匯出的物理群集邊界進行高精度標記
- **完整資訊**: 顯示群集ID、運動點數、檔案來源

### 階段 2: ROI 選擇與驗證
- **操作**: 滑鼠拖拽選擇 ROI 區域
- **最小尺寸**: 50×50 像素限制，確保有效標記空間
- **視覺回饋**: 紅色虛線框顯示選擇區域
- **座標轉換**: 自動處理畫布↔影像座標轉換
- **確認機制**: 按 `N` 進入精細標記模式

### 階段 3: 超高精度線段標記 (8倍放大)
- **8倍放大**: ROI區域8倍放大，極高精度標記
- **線段標記**: 點擊兩點形成參考線段（起點→終點）
- **多次標註系統**:
  - 支援無限次標註（超過3次會有提醒）
  - 按 `N` 進入下一步時才進行離群值剔除
  - 自動保留最接近平均值的3個標註
- **視覺標記**: 綠色起點 + 橙色終點 + 連線
- **操作快捷鍵**:
  - `R`: 重複標註當前線段
  - `Z`: 取消最後一次標註
  - `H`: 隱藏/顯示參考線段
  - `N`: 確認並進入下一階段

### 階段 4: 第二條線段標記
- **自動載入**: 載入 `post_cluster_XXX.jpg` 後零點參考
- **對比標記**: 顯示第一條線段供參考（青色線段）
- **相同精度**: 8倍放大 + 多次標註系統
- **離群值處理**: 按 `N` 時自動剔除兩條線段的離群標註
- **視覺區分**: 黃色線段區分第二條線段

### 階段 5: 智能位移計算與校正
- **平均計算**: 基於剔除離群值後的3次標註平均線段
- **Y分量計算**: `displacement = line2.y_component - line1.y_component`
- **比例換算**: `displacement_mm = (pixel_diff × 10.0) / scale_factor`
- **比較警示**: 人工值 < 程式估計值95% 時觸發警告對話框
- **三種選擇**:
  1. **使用程式估計值**: 採用自動檢測結果
  2. **重新標註**: 完全重置到第一條線段，清空所有標註
  3. **使用人工校正值**: 採用人工測量結果
- **比例分配**: 按原始值比例分配校正值到整個物理群集
- **雜訊過濾**: 小於閾值的位移自動歸零

### 階段 6: 智能儲存與暫存
- **完成檢測**: 自動檢測是否所有群集已處理完成
- **暫存選項**: 工作未完成時提供暫存功能
- **暫存格式**: JSON格式，包含時間戳、進度、標註記錄、CSV修改
- **最終儲存**: 統一使用 `mc` 前綴，支援覆蓋已有校正檔案

## 數據校正演算法

### 線段抗晃動原理
線段標記法能有效消除畫面晃動對位移測量的影響：

**晃動示例**：
```
線段1 (前零點): 起點(100,200) → 終點(100,250)  Y分量=50像素
線段2 (後零點): 起點(102,205) → 終點(102,258)  Y分量=53像素
```

雖然整體畫面向右上方移動了 (2,5) 像素，但：
- 實際位移 = 53 - 50 = 3像素 ✓ (晃動被抵消)
- 單點法誤差 = 258 - 250 = 8像素 ✗ (包含晃動)

### 多次標註平均系統
每條線段支援多次標註以提高精度：

**標註流程**：
1. 使用者可標註任意次數（建議3次）
2. 超過3次時按 `N` 進入下一步才剔除離群值
3. 迭代剔除離平均值最遠的標註，保留3個最接近的
4. 計算平均座標作為最終線段

**離群值剔除算法**：
```python
def remove_outliers(annotations, max_keep=3):
    while len(annotations) > max_keep:
        y_components = [line.y_component for line in annotations]
        mean_y = sum(y_components) / len(y_components)

        # 找到離平均最遠的標註
        max_distance = 0
        outlier_index = 0
        for i, y_comp in enumerate(y_components):
            distance = abs(y_comp - mean_y)
            if distance > max_distance:
                max_distance = distance
                outlier_index = i

        # 剔除離群值
        annotations.pop(outlier_index)
```

### 物理群集比例分配校正
基於物理群集邊界進行整體校正：

**物理群集範圍**：
```
pre_cluster_001.jpg  ← 前零點（標記第一條線段）
   第469項: 0        ← 物理群集起始
   第470項: 0.3mm    ← 運動開始
   第471項: 0.2mm    ← 運動中
   第472項: 0.5mm    ← 運動結束
   第473項: 0        ← 物理群集結束
post_cluster_001.jpg ← 後零點（標記第二條線段）
```

**校正範例**：
如果線段測量的實際位移是 4mm，整個物理群集按比例校正：
```
第469項: 0           (保持不變)
第470項: 4 × (0.3/1.0) = 1.2mm
第471項: 4 × (0.2/1.0) = 0.8mm
第472項: 4 × (0.5/1.0) = 2.0mm
第473項: 0           (保持不變)
```

### 智能校正公式
```python
def apply_physical_cluster_correction(physical_cluster, measured_displacement):
    """
    對整個物理群集區間應用校正

    Args:
        physical_cluster: 物理群集物件（包含前後零點範圍）
        measured_displacement: 基於線段平均的測量位移 (mm)

    Returns:
        bool: 是否成功應用校正（False表示視為雜訊）
    """
    # 雜訊檢測
    min_threshold = (10.0 / scale_factor) * 0.1
    if abs(measured_displacement) < min_threshold:
        # 整個物理群集設為零
        for i in range(physical_cluster.pre_zero_index,
                      physical_cluster.post_zero_index + 1):
            df.iloc[i, displacement_col_index] = 0.0
        return False

    # 獲取非零值並按比例分配
    non_zero_values = get_non_zero_values_in_cluster(physical_cluster)
    total_original = sum(abs(val) for val in non_zero_values)

    for idx, original_val in zip(non_zero_indices, non_zero_values):
        ratio = abs(original_val) / total_original
        corrected_val = measured_displacement * ratio
        # 保持原始正負號
        if original_val < 0:
            corrected_val = -corrected_val
        df.iloc[idx, displacement_col_index] = corrected_val

    return True
```

## 技術架構

### GUI 框架與功能
- **框架**: Tkinter (Python內建，跨平台相容)
- **核心功能**:
  - 影片幀顯示與縮放
  - 滑鼠拖拽ROI選擇
  - 精細線段標記
  - 8倍放大顯示
  - 完整鍵盤快捷鍵支援
  - 暫存檔案選擇對話框

### 檔案格式支援

#### 輸入格式
- **CSV檔案**: `*.csv` (智能欄位檢測)
  - 不帶前綴: `1.csv`, `2.csv`
  - 清理後: `c1.csv`, `c2.csv`
  - 已校正: `mc1.csv`, `mc2.csv`
  - **必需欄位**: 需要包含 `frame_path` 欄位（用於物理群集標籤）
  - **位移欄位**: 自動檢測以下順序 → `displacement` / `displacement_mm` / `位移` / `位移_mm` → 第3欄（按位置）
  - **時間軸欄位**: 需要 `second` 欄位（時戳）、`frame_idx` 欄位（幀號，可選用於精確提取）
- **影片檔案**: `lifts/data/*.mp4`
- **預匯出幀**: `lifts/exported_frames/{video_name}/*.jpg` (物理群集參考幀)
- **暫存檔案**: `{csv_name}_temp_{timestamp}.json`

#### 輸出格式
- **校正CSV**: 統一使用 `mc` 前綴
- **暫存JSON**: 包含完整工作狀態

### 核心模組架構

#### 1. 智能數據管理模組 (`DataManager`)
```python
class DataManager:
    def __init__(self, csv_path, video_name):
        # 智能欄位檢測與CSV載入
        self.displacement_column = self._find_displacement_column()
        self.displacement_col_index = self.df.columns.get_loc(self.displacement_column)

    def _find_displacement_column(self):
        # 智能找到位移欄位：'displacement', '位移', 或按位置

    def _identify_physical_clusters_from_png_tags(self):
        # 基於frame_path欄位識別物理群集

    def apply_physical_cluster_correction(self, physical_cluster, measured_displacement):
        # 物理群集整體校正
```

#### 2. JPG檔案處理模組 (`JPGHandler`)
```python
class JPGHandler:
    def __init__(self, video_name):
        # JPG處理器初始化（不需要視頻檔案）
        self.video_name = video_name
        self.rotation_angle = rotation_config.get(video_name, 0)

    def load_jpg_frame(self, jpg_filename):
        # 載入預匯出JPG參考幀
        # 應用旋轉校正（如果有設定）
```

#### 3. 高精度校正界面 (`CorrectionApp`)
```python
class CorrectionApp:
    def __init__(self, root, data_manager, jpg_handler):
        self.zoom_factor = 8  # 8倍放大精度
        self.max_annotations = 3  # 最多保留3次標註
        self.line_annotations = [[], []]  # 多次標註記錄

    def enter_precision_marking_mode(self):
        # 8倍放大ROI顯示

    def add_line_annotation(self, line):
        # 多次標註記錄（延遲剔除）

    def remove_outlier_annotations(self, line_index):
        # 批量離群值剔除

    def reset_to_first_line_annotation(self):
        # 重新標註完整重置

    def save_temporary_state(self):
        # 暫存系統

    def load_temporary_state(self, temp_data):
        # 狀態恢復
```

## 使用者介面設計

### 主視窗佈局
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 半自動位移校正工具 - 1.mp4 | 群集 3/15 | 幀號: 1500→1494 | 時戳: 46.9s     │
├──────────────────────────────────────────────────────────────────────────────┤
│ 檔案: 1.csv | 物理群集: 3/15 | ID: 003 | 運動點數: 8 | 使用JPG            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          影片畫面區域                                         │
│                    (ROI選擇/8倍放大標記)                                      │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ 階段2a: 8倍放大精細標記 - 請點擊第一條參考線段的起點 [已標註: 2/3] | 參考線段: 顯示 │
│ 快捷鍵: [N]ext [B]ack [S]ave [Q]uit [H]ide線段 [R]epeat [Z]取消            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 智能操作模式
1. **ROI 選擇模式**: 滑鼠拖拽選擇矩形區域
2. **8倍放大標記模式**: 超高精度線段標記
3. **多次標註模式**: 支援無限次標註，延遲剔除離群值
4. **暫存恢復模式**: 自動檢測並恢復工作狀態

### 視覺回饋系統
- **ROI 邊框**: 紅色虛線矩形 (5,5) dash pattern
- **線段標記**:
  - 綠色起點圓圈 + 白色十字
  - 橙色終點圓圈 + 白色十字
  - 青色第一條線段 (width=2)
  - 黃色第二條線段 (width=2)
  - 綠色連線 (width=6)
- **參考線段控制**: [H] 鍵切換顯示/隱藏
- **狀態指示**:
  - 頂部資訊欄：檔案、群集、幀號資訊
  - 底部狀態列：詳細操作提示和標註進度
  - 視窗標題：當前處理狀態

## 輸出格式

### 暫存檔案 (JSON)
工作未完成時自動生成，支援完整狀態恢復：
```json
{
    "metadata": {
        "csv_file": "1.csv",
        "csv_path": "D:/Lift_project/lifts/result/1.csv",
        "video_file": "1.mp4",
        "save_timestamp": "2024-12-15T14:30:25.123456",
        "format_version": "1.0"
    },
    "progress": {
        "current_cluster_index": 3,
        "total_clusters": 15,
        "current_phase": "line_marking_2",
        "current_line_index": 1,
        "current_point_in_line": 0
    },
    "settings": {
        "map_frames_enabled": false,
        "zoom_factor": 8,
        "max_annotations": 3
    },
    "current_state": {
        "roi_rect": [150, 200, 300, 250],
        "show_reference_lines": true,
        "line_annotations": [
            [
                {
                    "timestamp": 46.9,
                    "start_pixel_coords": [320, 240],
                    "end_pixel_coords": [320, 258],
                    "y_component": 18.0
                }
            ],
            []
        ]
    },
    "csv_modifications": {
        "completed_clusters": [
            {
                "cluster_index": 0,
                "physical_cluster_id": 1,
                "csv_row_range": [469, 473],
                "modified_values": {
                    "469": 0.0,
                    "470": 1.2,
                    "471": 0.8,
                    "472": 2.0,
                    "473": 0.0
                }
            }
        ]
    }
}
```

### 校正後CSV檔案
統一命名規則，支援覆蓋：
- **輸入**: `1.csv`, `c1.csv`, `mc1.csv`
- **輸出**: `mc1.csv` (統一使用mc前綴)
- **格式**: 保持原始CSV結構，僅更新位移欄位值

## 實作完成狀態

✅ **完整功能已實作並部署** (2024-12-15)

### 重大功能更新

#### 1. 智能檔案處理系統
- ✅ **多格式CSV支援**: `1.csv`, `c1.csv`, `mc1.csv` 全面兼容
- ✅ **智能欄位檢測**: 自動識別 `displacement`, `displacement_mm`, `位移`
- ✅ **統一輸出格式**: 所有校正結果統一使用 `mc` 前綴
- ✅ **向下兼容**: 支援新舊CSV格式，自動回退到位置檢測

#### 2. 高精度標註系統
- ✅ **8倍放大精度**: ROI區域8倍放大，極高精度標記
- ✅ **多次標註支援**: 無限次標註，延遲到按N時才剔除離群值
- ✅ **智能離群值處理**: 迭代剔除最遠離平均值的標註，保留最佳3個
- ✅ **視覺遮擋控制**: [H]鍵切換參考線段顯示，避免干擾標記

#### 3. 暫存恢復系統
- ✅ **智能暫存檢測**: 啟動時自動搜尋 `{name}_temp_{timestamp}.json`
- ✅ **多版本管理**: 支援多個暫存檔案選擇，顯示建立時間和進度
- ✅ **完整狀態恢復**: 包括進度、標註記錄、CSV修改、界面設定
- ✅ **工作流程保護**: 已完成群集數據完全不受重新標註影響

#### 4. JPG檔案必需系統
- ✅ **嚴格要求**: 所有物理群集必須擁有對應的JPG參考幀
- ✅ **清晰報錯**: 缺少JPG檔案時立即報錯，不進行任何回退操作
- ✅ **簡化流程**: 移除了幀映射和視頻幀回退的複雜邏輯
- ✅ **數據完整性**: 確保只使用經過驗證的高精度JPG參考幀

### 核心技術架構

#### 數據管理模組 (`DataManager`)
```python
# 智能欄位檢測
self.displacement_column = self._find_displacement_column()
self.displacement_col_index = self.df.columns.get_loc(self.displacement_column)

# 物理群集識別
self.physical_clusters = self._identify_physical_clusters_from_png_tags()

# 智能校正應用
def apply_physical_cluster_correction(self, physical_cluster, measured_displacement):
    # 雜訊檢測 + 比例分配 + 正負號保持
```

#### 高精度界面系統 (`CorrectionApp`)
```python
# 8倍放大系統
self.zoom_factor = 8
enlarged_roi = cv2.resize(roi_frame, None, fx=self.zoom_factor, fy=self.zoom_factor)

# 多次標註管理
self.line_annotations = [[], []]  # 兩條線段的多次標註記錄
self.max_annotations = 3

# 延遲離群值剔除
def remove_outlier_annotations(self, line_index):
    # 批量剔除直到保留最佳3個標註

# 暫存系統
def save_temporary_state(self) -> str:
    # 完整狀態序列化到JSON
def load_temporary_state(self, temp_data: dict):
    # 完整狀態恢復
```

### 使用方式

#### 啟動校正工具
```bash
cd src
uv run python manual_correction_tool.py
```

#### 完整工作流程
1. **智能檔案選擇**: 選擇任何格式的CSV檔案 (`1.csv`, `c1.csv`, `mc1.csv`)
2. **檔案驗證**: 檢查CSV是否包含 `frame_path` 欄位和物理群集資訊
3. **JPG驗證**: 確認所有物理群集JPG檔案已匯出到 `lifts/exported_frames/{video_name}/`
4. **暫存檢測**: 系統自動檢測暫存檔案，提供恢復選項
5. **狀態恢復**: 載入暫存狀態或從頭開始
6. **初始化JPG處理器**: 直接初始化JPG處理器（不需要檢查MP4檔案）
7. **JPG加載**: 加載預匯出JPG進行高精度標記（缺少JPG會報錯）
8. **ROI選擇**: 拖拽選擇包含參考特徵的區域
9. **8倍放大標記**: 超高精度多次線段標註
10. **智能校正**: 平均線段計算 + 位移比較警示
11. **智能儲存**: 工作未完成時提供暫存選項

#### 全面快捷鍵支援
- `N`: 進入下一步（執行離群值剔除）
- `B`: 返回上一步或上一個群集
- `S`: 智能儲存（完成時儲存CSV，未完成時提供暫存選項）
- `Q`: 退出應用程式
- `H`: 切換參考線段顯示/隱藏
- `R`: 重複標註當前線段
- `Z`: 取消最後一次標註

### 輸出格式

#### 暫存檔案
- **檔名**: `{csv_name}_temp_{timestamp}.json`
- **內容**: 完整工作狀態，包括進度、標註、CSV修改
- **恢復**: 自動檢測，多版本選擇

#### 最終CSV檔案
- **統一命名**: 所有輸入格式統一輸出為 `mc{name}.csv`
- **完整兼容**: 支援覆蓋已有校正檔案

### 技術突破

1. **智能檔案處理**: 完全消除檔案格式限制
2. **延遲離群值剔除**: 允許無限標註，提升標記靈活性
3. **完整狀態暫存**: 支援長期工作中斷和恢復
4. **嚴格必需驗證**: JPG檔案和CSV欄位必需，缺少立即報錯
5. **數據完整性保護**: 已完成工作絕對安全

### 必需配置文件

```python
# src/scale_config.py - 比例尺配置
scale_config = {
    "1.mp4": 150.5,      # 10mm 對應的像素數
    "2.mp4": 148.3,
    # ... 更多影片
}

# src/rotation_config.py - 旋轉校正配置
rotation_config = {
    "1.mp4": 0,          # 旋轉角度（度數），0=無旋轉
    "2.mp4": -90,        # -90=逆時針90度
    # ... 更多影片
}
```

### 必需CSV欄位

| 欄位名稱 | 型別 | 說明 | 可選性 |
|---------|------|------|--------|
| `frame_path` | str | 物理群集標籤（如 `pre_cluster_001.jpg`） | ❌ **必需** |
| `second` | float | 時間戳（秒） | ❌ 必需 |
| `frame_idx` | int | 影片幀號（用於顯示） | ✅ 可選 |
| `displacement` 或 `displacement_mm` 或 `位移` | float | 位移值 (mm) | ❌ 必需 |
| 其他欄位 | - | 自動保留 | ✅ 可選 |

**關鍵需求**:
- ⚠️ **缺少 `frame_path` 欄位時工具會報錯並停止執行**（物理群集標籤系統的核心需求）
- ⚠️ **缺少任何 JPG 檔案時工具會報錯並停止執行**（必須先執行 frame export 功能）
- ✅ `frame_idx` 欄位用於在控制台顯示幀號信息，不是必需的

### 依賴套件

```toml
dependencies = [
    "opencv-python",
    "numpy",
    "pandas",
    "pillow",
    "tkinter",  # Python內建
    "pathlib"   # Python內建
]
```

## 開發完成總結

半自動人工校正工具已達到生產就緒狀態，實現了以下重大突破：

### 🎯 **完整功能覆蓋**
- **多格式兼容**: 支援所有CSV格式，智能欄位檢測
- **暫存系統**: 完整的工作狀態保存和恢復
- **高精度標註**: 8倍放大 + 多次標註 + 智能離群值處理
- **智能校正**: 位移比較警示 + 完全重置機制

### 🚀 **使用者體驗優化**
- **一鍵啟動**: 自動檢測暫存，無縫恢復工作
- **視覺控制**: 參考線段顯示控制，避免標記干擾
- **靈活標註**: 無限次標註，延遲品質控制
- **智能儲存**: 自動判斷完成狀態，提供合適的儲存選項
- **清晰驗證**: 啟動時明確驗證必需的JPG檔案和CSV欄位，避免中途失敗
- **快速啟動**: 移除MP4檔案檢查，直接初始化JPG處理器，啟動更快

### 🛡️ **數據安全保障**
- **工作保護**: 已完成群集數據絕對安全
- **狀態隔離**: 重新標註只影響當前群集
- **嚴格驗證**: 缺少必需檔案/欄位時立即報錯，杜絕隱患

### 📈 **精度提升**
- **8倍放大**: 像素級精確定位
- **多次平均**: 自動剔除離群標註
- **物理群集**: 基於真實運動邊界的精確校正
- **比較警示**: 防止明顯測量錯誤

這個工具現在可以高效處理大規模的位移校正工作，為電梯位移檢測提供高品質的ground truth數據。
