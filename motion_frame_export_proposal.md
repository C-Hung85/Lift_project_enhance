### **專案企畫書：運動幀預儲存與校正流程優化**

#### 1. 專案名稱
運動偵測流程優化：關鍵幀預儲存機制與校正邏輯改進

#### 2. 專案目標
本專案包含兩個核心改進：

**A. 關鍵幀預儲存機制**
修改主偵測程式 `src/lift_travel_detection.py`，使其在偵測到運動時自動匯出關鍵影片幀為 PNG 圖片檔案，為手動校正工具提供 100% 準確的圖像來源，根本性解決幀導航偏差問題。

**B. 手動校正邏輯優化**
改進手動校正工具的群集邊界選擇策略，從「群集前0點 + 群集末點」改為「群集前0點 + 群集後0點」，更準確反映運動的真實開始和結束狀態。

#### 3. 背景與動機

**A. 幀導航偏差問題**
根據先前的調查報告 (`investigation_frame_navigation.md`)，手動校正工具在跳轉到特定 `frame_idx` 時存在顯著偏差，源於兩個模組採用不同的影片幀存取機制。

**B. 群集邊界選擇不當**
目前手動校正工具使用「群集末點」作為運動結束參考，但由於採樣間隔為6幀，群集末點可能仍有未檢測到的微量運動。「群集後0點」更能準確代表運動完全停止的狀態。

**C. 計算資源浪費**
目前程式在暗房區間仍進行完整的運動計算後才忽略結果，造成大量不必要的計算開銷。

**D. 儲存成本大幅優化**
基於物理群集標籤的新設計，每個物理群集僅需2張PNG（前0點+後0點），預估每個專案僅增加約 **20 MB**，45個專案總成本約 **900 MB**，相比原估算降低93%。

#### 4. 詳細實作計畫

##### 4.1. 物理群集標籤與PNG儲存架構

**A. CSV欄位標籤系統**
在CSV中新增 `frame_path` 欄位，通過標籤標識物理群集邊界：

```csv
frame_idx,second,vertical_travel_distance (mm),frame_path
6850,114.7,0.0,
6852,114.8,0.0,pre_cluster_001.png          # 物理群集1前0點
6858,115.1,2.1,
6864,115.4,0.0,                             # 雜訊清理插入的0點
6870,115.7,1.8,
6876,116.0,0.0,post_cluster_001.png         # 物理群集1後0點
6888,116.6,0.0,
6900,117.2,0.0,pre_cluster_002.png          # 物理群集2前0點（純雜訊）
6906,117.5,0.0,                             # 被清理的雜訊
6912,117.8,0.0,                             # 被清理的雜訊
6918,118.1,0.0,post_cluster_002.png         # 物理群集2後0點（純雜訊）
```

**B. 物理群集的兩種模式**

**模式1：真實運動群集**（需要手動校正）
- 前0點與後0點之間**包含非零值**
- 即使中間被雜訊清理分割，仍視為單一物理運動

**模式2：純雜訊群集**（自動跳過）
- 前0點與後0點之間**完全為0**（所有運動都被雜訊清理移除）
- 代表原本就是畫面抖動雜訊，不需要手動校正

**C. 目錄結構與命名規則**
```
lifts/
└── exported_frames/
    ├── 1/
    │   ├── pre_cluster_001.png    # 物理群集1前0點
    │   ├── post_cluster_001.png   # 物理群集1後0點
    │   ├── pre_cluster_002.png    # 物理群集2前0點（可能是純雜訊）
    │   └── post_cluster_002.png   # 物理群集2後0點
    ├── 21/
    │   ├── pre_cluster_001.png
    │   ├── post_cluster_001.png
    │   └── ...
    └── ...
```

**檔案命名規則**：
- **群集前0點**：`pre_cluster_{序號:03d}.png`
- **群集後0點**：`post_cluster_{序號:03d}.png`
- **序號**：從001開始，按物理群集檢測順序編號

##### 4.2. 修改 `src/lift_travel_detection.py`

**A. 暗房區間處理優化（性能關鍵改進）**

當前問題：程式在暗房區間仍執行完整運動計算（第114-169行），然後才將結果設為0（第175-176行），造成大量資源浪費。

**修改策略**：
1. **提前檢查暗房區間**：在第114行 `if ret and frame_idx % FRAME_INTERVAL == 0:` 之後立即檢查
2. **早期跳過**：如果在暗房區間，跳過所有運動計算，直接記錄零值結果
3. **保留必要資訊**：維持幀號更新以確保後續處理正確

**修改位置**：`scan` 函式第114-176行區間
```python
if ret and frame_idx % FRAME_INTERVAL == 0:
    # 提前檢查暗房區間
    current_time_seconds = frame_idx / fps
    is_darkroom, darkroom_info = is_in_darkroom_interval(current_time_seconds, darkroom_intervals_seconds)

    if is_darkroom:
        # 暗房區間：跳過所有計算，直接記錄零值
        result['frame'].append(frame)
        result['frame_idx'].append(frame_idx)
        result['keypoints'].append([])
        result['kp_pair_lines'].append([])
        result['camera_pan'].append(True)  # 標記為類似camera_pan
        result['v_travel_distance'].append(0)
        result['frame_path'].append('')  # 無匯出圖片
        continue

    # 原有的運動計算邏輯...
```

**B. 物理群集檢測與PNG匯出機制**

**核心設計**：
1. **物理群集狀態追蹤**：維持當前物理群集的狀態（序號、開始點）
2. **即時標籤記錄**：在CSV中即時標記前0點和後0點
3. **延遲匯出**：物理群集結束時匯出標記的PNG檔案

**新增變數**：
```python
# 在scan函式開始處添加
physical_cluster_counter = 0      # 物理群集序號計數器
in_physical_cluster = False       # 是否在物理群集中
current_cluster_id = None         # 當前物理群集ID
frame_cache = []                  # 緩存最近幀：[(frame_idx, frame), ...]
pending_pre_export = None         # 待匯出的前0點幀
```

**物理群集檢測與標籤邏輯**：
```python
def process_motion_detection(frame_idx, frame, vertical_travel_distance):
    """處理運動檢測並標記物理群集"""
    global physical_cluster_counter, in_physical_cluster, current_cluster_id
    global pending_pre_export, frame_cache

    # 維護幀緩存（保留最近20幀）
    frame_cache.append((frame_idx, frame.copy()))
    if len(frame_cache) > 20:
        frame_cache.pop(0)

    frame_path = ''  # 默認空標籤

    if vertical_travel_distance != 0 and not in_physical_cluster:
        # 開始新的物理群集
        physical_cluster_counter += 1
        current_cluster_id = physical_cluster_counter
        in_physical_cluster = True

        # 標記前一幀為前0點
        if len(result['frame_path']) > 0:
            result['frame_path'][-1] = f'pre_cluster_{current_cluster_id:03d}.png'
            # 記錄待匯出的前0點
            if frame_cache and len(frame_cache) >= 2:
                pending_pre_export = (frame_cache[-2], f'pre_cluster_{current_cluster_id:03d}.png')

    elif vertical_travel_distance == 0 and in_physical_cluster:
        # 物理群集結束，標記當前幀為後0點
        frame_path = f'post_cluster_{current_cluster_id:03d}.png'
        in_physical_cluster = False

        # 匯出前0點和後0點PNG
        if pending_pre_export:
            export_frame_png(*pending_pre_export)
            pending_pre_export = None

        export_frame_png((frame_idx, frame), frame_path)
        current_cluster_id = None

    result['frame_path'].append(frame_path)

def export_frame_png(frame_data, png_filename):
    """匯出單個幀為PNG"""
    frame_idx, frame = frame_data
    video_name = os.path.splitext(file_name)[0]

    export_path = f"lifts/exported_frames/{video_name}/{png_filename}"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    # 匯出原始未處理的幀
    cv2.imwrite(export_path, frame)
    print(f"📸 匯出PNG: {png_filename}")
```

**C. CSV欄位更新**

在 `result` 字典中新增 `frame_path` 欄位：
```python
result = {
    'frame':[],
    'frame_idx':[],
    'keypoints':[],
    'camera_pan':[],
    'v_travel_distance':[],
    'kp_pair_lines':[],
    'frame_path':[]     # 新增：對應的匯出圖片路徑
}
```

##### 4.3. 修改 `src/manual_correction_tool.py`

**A. 基於PNG標籤的物理群集識別**

完全重新設計群集識別邏輯，基於CSV中的PNG標籤而非運動值分析：

```python
@dataclass
class PhysicalCluster:
    """物理群集數據結構"""
    cluster_id: int                    # 物理群集序號
    pre_zero_index: int                # 前0點CSV行號
    post_zero_index: int               # 後0點CSV行號
    pre_zero_png: str                  # 前0點PNG檔名
    post_zero_png: str                 # 後0點PNG檔名
    region_values: List[float]         # 區間內的所有位移值
    is_pure_noise: bool                # 是否為純雜訊群集（區間內全為0）
    has_real_motion: bool              # 是否包含真實運動

def _identify_physical_clusters_from_png_tags(self) -> List[PhysicalCluster]:
    """基於PNG標籤識別物理群集 - 極其簡化的邏輯"""
    physical_clusters = []

    # 尋找所有前0點標籤
    for i, row in self.df.iterrows():
        frame_path = row.get('frame_path', '')

        if frame_path.startswith('pre_cluster_'):
            # 提取群集序號
            cluster_id = int(frame_path.split('_')[2].split('.')[0])

            # 找到對應的後0點
            post_tag = f'post_cluster_{cluster_id:03d}.png'
            post_rows = self.df[self.df['frame_path'] == post_tag]

            if not post_rows.empty:
                pre_zero_index = i
                post_zero_index = post_rows.index[0]

                # 分析區間內的運動值
                displacement_col = self.df.columns[2]  # displacement column
                region_values = self.df.iloc[pre_zero_index:post_zero_index+1][displacement_col].tolist()

                # 檢查是否為純雜訊群集
                non_zero_values = [v for v in region_values if v != 0]
                is_pure_noise = len(non_zero_values) == 0
                has_real_motion = not is_pure_noise

                cluster = PhysicalCluster(
                    cluster_id=cluster_id,
                    pre_zero_index=pre_zero_index,
                    post_zero_index=post_zero_index,
                    pre_zero_png=frame_path,
                    post_zero_png=post_tag,
                    region_values=region_values,
                    is_pure_noise=is_pure_noise,
                    has_real_motion=has_real_motion
                )

                # 只加入有真實運動的群集到校正清單
                if has_real_motion:
                    physical_clusters.append(cluster)
                    print(f"✅ 識別物理群集 {cluster_id}：包含 {len(non_zero_values)} 個運動點")
                else:
                    print(f"⚠️  跳過純雜訊群集 {cluster_id}：區間內無真實運動")

    print(f"📊 總共識別 {len(physical_clusters)} 個需要校正的物理群集")
    return physical_clusters
```

**B. 簡化的PNG載入邏輯**

```python
def load_cluster_reference_frames(self, cluster: PhysicalCluster):
    """載入物理群集的前0點和後0點PNG"""
    video_name = os.path.splitext(self.video_handler.video_name)[0]
    frames_dir = f"lifts/exported_frames/{video_name}"

    # 載入前0點PNG
    pre_png_path = os.path.join(frames_dir, cluster.pre_zero_png)
    pre_frame = cv2.imread(pre_png_path)

    if pre_frame is None:
        raise FileNotFoundError(f"找不到前0點PNG: {pre_png_path}")

    # 載入後0點PNG
    post_png_path = os.path.join(frames_dir, cluster.post_zero_png)
    post_frame = cv2.imread(post_png_path)

    if post_frame is None:
        raise FileNotFoundError(f"找不到後0點PNG: {post_png_path}")

    print(f"✅ 載入物理群集 {cluster.cluster_id} 的PNG檔案")
    return pre_frame, post_frame

def show_current_physical_cluster(self):
    """顯示當前物理群集的參考幀"""
    cluster = self.physical_clusters[self.current_cluster_index]

    if self.current_phase in ["roi_selection", "line_marking_1"]:
        # 第一條線段：前0點
        pre_frame, _ = self.load_cluster_reference_frames(cluster)
        self.show_frame(pre_frame)
        description = f"物理群集 {cluster.cluster_id} 前0點 (運動前狀態)"

    elif self.current_phase == "line_marking_2":
        # 第二條線段：後0點
        _, post_frame = self.load_cluster_reference_frames(cluster)
        self.show_frame(post_frame)
        description = f"物理群集 {cluster.cluster_id} 後0點 (運動後狀態)"

    # 更新資訊顯示
    total_clusters = len(self.physical_clusters)
    cluster_info = f"物理群集: {self.current_cluster_index + 1}/{total_clusters} | "
    cluster_info += f"ID: {cluster.cluster_id} | {description}"
    cluster_info += f" | 運動點數: {len([v for v in cluster.region_values if v != 0])}"

    self.info_label.config(text=cluster_info)
```

**C. 位移校正的區間處理**

```python
def apply_physical_cluster_correction(self, cluster: PhysicalCluster, measured_displacement: float):
    """對整個物理群集區間應用校正"""
    displacement_col = self.df.columns[2]

    # 獲取區間內所有非零值的位置和值
    region_start = cluster.pre_zero_index
    region_end = cluster.post_zero_index

    non_zero_indices = []
    non_zero_values = []

    for i in range(region_start, region_end + 1):
        value = self.df.iloc[i, 2]  # displacement column
        if value != 0:
            non_zero_indices.append(i)
            non_zero_values.append(value)

    if not non_zero_values:
        print("⚠️  警告：物理群集區間內無非零值")
        return False

    # 按比例分配校正值
    total_original = sum(abs(val) for val in non_zero_values)
    if total_original == 0:
        return False

    for idx, original_val in zip(non_zero_indices, non_zero_values):
        ratio = abs(original_val) / total_original
        corrected_val = measured_displacement * ratio

        # 保持原始正負號
        if original_val < 0:
            corrected_val = -corrected_val

        self.df.iloc[idx, 2] = corrected_val

    print(f"✅ 物理群集 {cluster.cluster_id} 校正完成：{len(non_zero_indices)} 個點")
    return True
```

#### 5. 實作順序與階段劃分

**第一階段：基礎架構與暗房優化**
1. **暗房區間處理優化**（優先級：高，風險：低）
   - 修改 `scan` 函式的主迴圈邏輯
   - 提前暗房檢查，跳過不必要計算
   - 預期性能提升：暗房區間處理速度提升50-80%

2. **CSV欄位結構更新**（優先級：高，風險：低）
   - 在 `result` 字典中新增 `frame_path` 欄位
   - 更新CSV輸出邏輯
   - 確保向後兼容性

**第二階段：群集緩存機制**
3. **幀緩存系統實作**（優先級：中，風險：中）
   - 實作有限大小的幀緩存（20幀）
   - 群集狀態追蹤變數
   - 緩存管理與記憶體控制

4. **群集檢測與匯出邏輯**（優先級：中，風險：中）
   - 運動群集的開始/結束檢測
   - 群集前0點和群集後0點的識別
   - PNG檔案匯出機制

**第三階段：手動校正工具改進**
5. **群集識別邏輯更新**（優先級：高，風險：中）
   - 修改 `_identify_clusters` 函式
   - 新增 `post_zero_index` 支援
   - 更新 `CorrectionCluster` 資料結構

6. **參考幀選擇邏輯**（優先級：高，風險：低）
   - 修改 `show_current_cluster` 函式
   - 實現「群集前0點 + 群集後0點」策略
   - 更新GUI顯示邏輯

**第四階段：整合測試與優化**
7. **PNG載入邏輯**（優先級：中，風險：低）
   - 實作PNG優先載入機制
   - 影片回退邏輯
   - 錯誤處理與重試機制

8. **完整性測試**（優先級：高，風險：低）
   - 端到端測試流程
   - 性能基準測試
   - 相容性驗證

#### 6. 預期效益與量化指標

**A. 準確性提升**
- **幀導航精度**：從85-90%提升至100%（基於PNG完全匹配）
- **物理群集識別**：基於主程式標籤而非後處理分析，準確性100%
- **群集邊界精度**：使用「物理群集前0點 + 物理群集後0點」，準確反映真實運動邊界
- **校正一致性**：消除手動校正中的幀導航變異性和群集分割問題

**B. 性能優化**
- **暗房區間處理**：CPU使用率降低50-80%（跳過特徵檢測和匹配）
- **I/O優化**：PNG載入速度比影片隨機存取快3-5倍
- **記憶體使用**：幀緩存限制在20幀內，記憶體占用可控
- **儲存優化**：從14GB降低至900MB（93%成本節省）

**C. 使用者體驗**
- **操作簡化**：移除 `--map-frames` 參數需求，減少用戶錯誤
- **視覺一致性**：GUI顯示與主程式分析完全一致
- **工作流暢度**：物理群集邊界選擇更符合物理直覺
- **智能過濾**：自動跳過純雜訊群集，只校正真實運動

#### 7. 風險評估與緩解措施

**A. 技術風險**

| 風險項目 | 機率 | 影響 | 緩解措施 |
|---------|------|------|----------|
| 幀緩存記憶體溢出 | 低 | 中 | 限制緩存大小20幀，實作LRU清理 |
| PNG匯出I/O性能影響 | 低 | 低 | 每群集僅2張PNG，異步寫入優化 |
| 物理群集標籤錯誤 | 中 | 高 | 詳細單元測試，邊界條件驗證 |
| CSV格式相容性問題 | 低 | 中 | 保持向後相容，添加版本檢查 |
| PNG檔案遺失或損壞 | 低 | 中 | 檔案完整性檢查，回退到影片載入 |

**B. 資源風險**

| 風險項目 | 影響評估 | 緩解措施 |
|---------|----------|----------|
| 儲存空間需求 | 每專案+20MB | 93%成本節省，可接受範圍 |
| 開發時間延長 | +2-3週開發週期 | 階段性交付，增量測試 |
| 向後相容性 | 舊版工具不適用 | 保留舊版支援，漸進遷移 |
| PNG標籤系統複雜性 | 新增標籤邏輯 | 詳細文檔，單元測試覆蓋 |

**C. 操作風險**
- **使用者適應期**：物理群集概念需要用戶理解，但邏輯更直觀
- **數據遷移**：現有專案需要重新執行主程式以產生PNG檔案和標籤
- **工具鏈耦合**：手動校正工具依賴PNG檔案，但提供回退機制

#### 8. 測試與驗證計畫

**A. 單元測試**
- 物理群集標籤邏輯測試（邊界條件、純雜訊群集、複合群集）
- 幀緩存機制測試（溢出處理、LRU機制）
- PNG匯出功能測試（檔案完整性、路徑處理、序號命名）
- CSV標籤解析測試（格式驗證、錯誤處理）

**B. 整合測試**
- 主程式與手動校正工具的端到端測試
- 不同影片格式和配置的相容性測試
- 性能基準測試（處理時間、記憶體使用）

**C. 用戶驗收測試**
- 使用代表性數據集進行完整校正流程測試
- 準確性對比測試（新舊方法對比）
- 使用者體驗評估

**D. 壓力測試**
- 長時間連續處理測試
- 大檔案和高幀率影片測試
- 記憶體洩漏和資源回收測試

#### 9. 部署與回滾計畫

**A. 漸進式部署**
1. **測試環境驗證**：在測試環境完成所有測試
2. **小規模試點**：選擇3-5個代表性專案進行試運行
3. **逐步推廣**：根據試點結果逐步擴大應用範圍

**B. 回滾策略**
- **程式碼版本**：使用Git分支管理，確保可快速回滾
- **數據備份**：處理前自動備份原始數據
- **相容模式**：保留舊版本功能作為備用選項

#### 10. 成功指標

**量化指標**：
- 手動校正準確性提升：≥15%
- 暗房區間處理性能提升：≥50%
- 幀導航偏差消除：100%
- 使用者操作錯誤減少：≥30%

**質化指標**：
- 使用者滿意度調查
- 系統穩定性評估
- 維護成本評估

透過此全面的改進方案，我們將顯著提升整個運動偵測和校正系統的準確性、效率和使用者體驗。
