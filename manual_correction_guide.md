# 半自動人工校正工作流開發指引

## 概要

這個工作流程用於手動校正電梯位移檢測數據，通過視覺化的方式讓使用者標記真實的位移參考點，以提供 ground truth 來校正自動檢測的結果。

## 系統架構

### 核心模組
- **影片處理**: `lifts/data/*.mp4` - 原始影片檔案
- **旋轉校正**: `src/rotation_config.py` + `src/rotation_utils.py` - 影片旋轉處理
- **比例尺換算**: `src/scale_config.py` - 像素到毫米的換算係數
- **清理數據**: `lifts/result/c*.csv` - 經過雜訊清理的位移數據

### 工作流程圖
```
原始影片 → 旋轉校正 → 幀提取 → ROI選擇 → 參考點標記 → 位移校正 → 更新CSV
    ↓           ↓         ↓        ↓         ↓         ↓          ↓
  data/     rotation   frame    user     ground    scale    corrected
 *.mp4      utils     by time   ROI     truth    conversion   CSV
```

## 詳細工作流程

### 階段 0: 初始化
1. 載入原始影片檔案 (`lifts/data/*.mp4`)
2. 檢查是否需要旋轉校正 (`rotation_config.py`)
3. 如需旋轉，使用 `rotation_utils.py` 進行校正
4. 載入對應的清理後 CSV 檔案 (`lifts/result/c*.csv`)
5. 載入比例尺配置 (`scale_config.py`)

### 階段 1: 導航到第一個校正點
- **目標**: 找到第一個非零值的前一個採樣點
- **條件**: 該時戳的位移應該是 0
- **範例**: 如果 CSV 中第 469 項是 0，第 470 項是第一個非零值，則導航到第 469 項對應的影片時戳

### 階段 2: ROI 選擇
- **操作**: 使用者在完整畫面上框選 ROI (Region of Interest)
- **目的**: 選擇包含絕對參考點的區域
- **介面**: 滑鼠拖拽矩形框選
- **確認**: 按下確認鍵進入下一階段

### 階段 3: 精細標記 (4倍放大)
- **顯示**: 僅顯示 ROI 區域，放大 4 倍
- **標記工具**: 細十字線 (線寬 = 4 像素，相當於原檔案 1 像素)
- **操作**: 滑鼠點擊標記絕對參考點位置
- **確認**: 按下 'n' (next) 確認並記錄座標

### 階段 4: 移動到群集結束點
- **邏輯**: 跳轉到當前非零值群集的最後一個採樣點
- **範例**: 
  ```
  第469項: 0     ← 第一次標記點
  第470項: 0.3   ← 群集開始
  第471項: 0.2   
  第472項: 0.5   ← 群集結束，第二次標記點
  第473項: 0     ← 下一群集的起始前點
  ```
- **操作**: 使用者再次進行 ROI 選擇和精細標記
- **確認**: 按下 'n' 記錄第二個參考點

### 階段 5: 位移計算與校正
- **計算**: 兩個參考點的 Y 軸差值 (像素)
- **換算**: 使用 `scale_config.py` 中的比例係數轉換為毫米
- **公式**: `實際位移(mm) = Y差值(像素) × 10 / scale_config[影片名稱]`
- **校正**: 按比例分配校正整個群集的位移值

### 階段 6: 迭代處理
- **繼續**: 移動到下一個非零群集的前一個零點
- **重複**: 階段 2-5，直到處理完所有非零群集
- **完成**: 儲存校正後的 CSV 檔案

## 數據校正演算法

### 比例分配校正
假設原始清理後的數據：
```
第469項: 0
第470項: 0.3   (比例: 0.3/1.0 = 30%)
第471項: 0.2   (比例: 0.2/1.0 = 20%)  
第472項: 0.5   (比例: 0.5/1.0 = 50%)
第473項: 0
```

如果測量的實際位移是 4mm，校正後：
```
第469項: 0
第470項: 4 × 0.3 = 1.2mm
第471項: 4 × 0.2 = 0.8mm
第472項: 4 × 0.5 = 2.0mm
第473項: 0
```

### 校正公式
```python
def correct_cluster(cluster_values, measured_displacement):
    """
    校正一個位移群集的值
    
    Args:
        cluster_values: 原始群集位移值列表
        measured_displacement: 測量的真實位移 (mm)
    
    Returns:
        corrected_values: 校正後的位移值列表
    """
    total_original = sum(abs(val) for val in cluster_values if val != 0)
    if total_original == 0:
        return cluster_values
    
    corrected_values = []
    for val in cluster_values:
        if val == 0:
            corrected_values.append(0)
        else:
            # 按原始值的比例分配測量位移
            ratio = abs(val) / total_original
            corrected_val = measured_displacement * ratio
            # 保持原始正負號
            if val < 0:
                corrected_val = -corrected_val
            corrected_values.append(corrected_val)
    
    return corrected_values
```

## 技術需求

### GUI 框架
- **推薦**: Tkinter 或 PyQt5/6
- **需求**: 
  - 影片播放控制
  - 自定義畫布繪製
  - 滑鼠事件處理
  - 鍵盤快捷鍵

### 核心功能模組

#### 1. 影片處理模組
```python
class VideoHandler:
    def __init__(self, video_path, rotation_angle=0):
        # 載入影片，應用旋轉
    
    def get_frame_at_time(self, timestamp):
        # 獲取指定時戳的影片幀
    
    def apply_rotation(self, frame):
        # 應用旋轉校正
```

#### 2. 數據管理模組
```python
class DataManager:
    def __init__(self, csv_path, scale_factor):
        # 載入清理後的 CSV 和比例尺
    
    def find_non_zero_clusters(self):
        # 識別所有非零值群集
    
    def get_correction_points(self):
        # 獲取需要校正的時戳點
    
    def apply_correction(self, cluster_idx, measured_displacement):
        # 應用校正到指定群集
```

#### 3. GUI 控制模組
```python
class CorrectionGUI:
    def __init__(self, video_handler, data_manager):
        # 初始化界面
    
    def show_full_frame(self, timestamp):
        # 顯示完整幀供 ROI 選擇
    
    def show_roi_zoom(self, roi_rect, zoom_factor=4):
        # 顯示放大的 ROI 區域
    
    def handle_roi_selection(self, event):
        # 處理 ROI 選擇事件
    
    def handle_reference_point_click(self, event):
        # 處理參考點標記事件
```

## 使用者介面設計

### 主視窗佈局
```
┌─────────────────────────────────────────────────────────────┐
│ 檔案: 1.mp4 | 時戳: 46.9s | 群集: 1/15 | 模式: ROI選擇      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                    影片畫面區域                              │
│                  (ROI選擇/放大顯示)                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ 狀態: 請框選包含參考點的 ROI 區域                            │
│ 快捷鍵: [N]ext  [B]ack  [S]ave  [Q]uit                     │
└─────────────────────────────────────────────────────────────┘
```

### 操作模式
1. **ROI 選擇模式**: 拖拽選擇矩形區域
2. **精細標記模式**: 點擊標記參考點 (4倍放大)
3. **確認模式**: 按 'n' 確認並進入下一個點

### 視覺回饋
- **ROI 邊框**: 紅色虛線矩形
- **參考點標記**: 綠色十字線 (4像素線寬)
- **狀態指示**: 底部狀態列顯示當前操作提示
- **進度指示**: 頂部顯示當前群集進度

## 輸出格式

### 校正記錄檔案
儲存為 JSON 格式，記錄每次校正的詳細資訊：
```json
{
    "video_file": "1.mp4",
    "csv_file": "c1.csv",
    "scale_factor": 86.2,
    "corrections": [
        {
            "cluster_index": 0,
            "start_timestamp": 46.9,
            "end_timestamp": 47.3,
            "reference_points": [
                {"timestamp": 46.9, "pixel_coords": [320, 240]},
                {"timestamp": 47.3, "pixel_coords": [320, 258]}
            ],
            "measured_displacement_mm": 2.1,
            "original_values": [0, 0.3, 0.2, 0.5, 0],
            "corrected_values": [0, 1.26, 0.84, 2.1, 0]
        }
    ],
    "completion_time": "2025-09-21T15:30:00",
    "total_clusters": 15,
    "corrected_clusters": 8
}
```

### 校正後的 CSV 檔案
在原檔名基礎上加上 `_corrected` 後綴：
- `c1.csv` → `c1_corrected.csv`
- 保持原始格式，僅更新位移值

## 開發優先順序

### Phase 1: 核心功能 (必須)
1. 影片載入與旋轉校正
2. CSV 數據讀取與群集識別
3. 基本 GUI 框架與影片顯示
4. ROI 選擇功能

### Phase 2: 標記功能 (必須)
1. 放大顯示 ROI
2. 精細十字線標記
3. 座標記錄與計算
4. 基本位移校正

### Phase 3: 完整工作流 (必須)
1. 群集間導航
2. 批量處理邏輯
3. 數據儲存功能
4. 錯誤處理

### Phase 4: 改善功能 (建議)
1. 撤銷/重做功能
2. 快捷鍵優化
3. 批次處理多檔案
4. 品質檢查工具

## 注意事項

### 數據一致性
- 確保時戳與影片幀的準確對應
- 驗證比例尺的正確性
- 檢查旋轉校正的影響

### 使用者體驗
- 提供清晰的操作指引
- 實作快捷鍵支援
- 顯示進度和狀態資訊
- 支援操作撤銷

### 效能考量
- 影片幀的快取機制
- ROI 區域的即時渲染
- 大型 CSV 檔案的處理

### 錯誤處理
- 檔案缺失或損壞
- 比例尺配置錯誤
- 使用者操作錯誤
- 數據格式不匹配

## 實作完成狀態

✅ **核心功能已完成實作** (2025-09-21)

### 已實作的功能模組

#### 1. 數據管理模組 (`DataManager`)
- ✅ CSV 檔案讀取與解析
- ✅ 非零值群集自動識別
- ✅ 比例尺配置載入 (`scale_config.py`)
- ✅ 位移校正演算法與比例換算
- ✅ 雜訊檢測與過濾 (小於0.5像素閾值)
- ✅ 校正後數據輸出為 `mc*.csv` 格式

#### 2. 影片處理模組 (`VideoHandler`)
- ✅ 影片檔案載入與播放控制
- ✅ 基於時戳的精確幀提取
- ✅ 旋轉校正整合 (`rotation_config.py` + `rotation_utils.py`)
- ✅ 座標系統轉換處理

#### 3. GUI 校正界面 (`CorrectionApp`)
- ✅ 完整的 Tkinter 圖形界面
- ✅ ROI 拖拽選擇功能
- ✅ 4倍放大精細標記模式
- ✅ 綠色十字線標記 (4像素線寬對應原影像1像素)
- ✅ 鍵盤快捷鍵支援 ([N]ext, [B]ack, [S]ave, [Q]uit)
- ✅ 群集間自動導航
- ✅ 實時狀態指示與進度顯示

### 使用方式

#### 啟動校正工具
```bash
cd src
uv run python manual_correction_tool.py
```

#### 工作流程
1. **選擇檔案**: 選擇清理後的 CSV 檔案 (如 `c1.csv`)
2. **自動載入**: 系統自動載入對應的影片檔案和配置
3. **ROI 選擇**: 在第一個標記點拖拽選擇 ROI 區域
4. **精細標記**: 在4倍放大視圖中點擊標記參考點
5. **確認標記**: 按 `N` 確認並移至第二個標記點
6. **自動校正**: 完成兩點標記後自動計算並應用校正
7. **繼續處理**: 自動移至下一個群集，重複流程
8. **儲存結果**: 完成後按 `S` 儲存為 `mc*.csv` 檔案

#### 快捷鍵操作
- `N` (Next): 確認當前標記，進入下一步
- `B` (Back): 返回上一步或上一個群集
- `S` (Save): 儲存校正結果
- `Q` (Quit): 退出應用程式

### 輸出格式

校正後的檔案命名規則：
- 輸入: `c1.csv` (清理後)
- 輸出: `mc1.csv` (清理+手動校正後)

### 技術特性

#### 精度控制
- **座標精度**: 像素級別精確定位
- **線寬設計**: 4像素線寬對應原影像1像素
- **雜訊過濾**: 小於0.5像素位移自動視為雜訊

#### 座標系統
- **Y軸方向**: 向上為正 (符合物理定義)
- **旋轉處理**: 剛性旋轉，保持尺寸和比例
- **多重座標**: 畫布↔ROI↔原影像 座標自動轉換

#### 比例換算
```python
# 位移計算公式
pixel_diff_y = point1.y - point2.y  # 向上為正
displacement_mm = (pixel_diff_y * 10.0) / scale_factor
```

#### 校正演算法
```python
# 比例分配校正
for original_val in cluster:
    ratio = abs(original_val) / total_original
    corrected_val = measured_displacement * ratio
    if original_val < 0:
        corrected_val = -corrected_val
```

### 錯誤處理

- ✅ 檔案缺失檢測
- ✅ 比例尺配置驗證  
- ✅ 影片幀提取失敗處理
- ✅ ROI 大小驗證 (最小50×50像素)
- ✅ 座標邊界檢查
- ✅ 雜訊位移自動過濾
- ✅ **設備故障檢測** - 第一行即有位移的智能處理

### 設備故障處理

當檢測到檔案從第一行就開始有位移時（通常表示設備故障），系統會：

1. **自動識別**: 檢測無前零點的異常群集
2. **顯示畫面**: 展示故障發生時的影片幀
3. **提供選項**: 
   - **清零**: 將整個故障群集設為 0（推薦）
   - **跳過**: 保持原值但不進行校正
   - **檢視**: 返回畫面進一步檢查
4. **繼續流程**: 自動移至下一個正常群集

這確保了即使遇到設備故障也能繼續完成校正工作。

### 依賴套件

```toml
dependencies = [
    "opencv-python",
    "numpy", 
    "pandas",
    "pillow",
    "tkinter"  # 通常內建於Python
]
```

這個工作流程將大幅提升位移檢測的準確性，通過人工校正提供高品質的 ground truth 數據。

## 開發完成總結

半自動人工校正工具已完整實作並可投入使用。主要成果：

1. **完整GUI工作流**: 從檔案選擇到結果輸出的全自動流程
2. **高精度標記**: 4倍放大+像素級精確定位
3. **智能校正**: 自動比例分配+雜訊過濾
4. **使用者友善**: 直觀的視覺界面+快捷鍵操作
5. **數據一致性**: 完整的座標轉換+配置整合

使用者現在可以高效地進行位移數據的手動校正，大幅提升檢測精度。
