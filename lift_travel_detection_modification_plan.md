## 目的

記錄 `src/lift_travel_detection.py` 對「物理運動群集檢測」的修改計畫，避免重蹈先前「非零點數遠少於群集數」的邏輯錯誤。此計畫僅調整偵測的時序與狀態處理，保留現有 ORB/KMeans/t-test 與 camera pan 排除的核心流程。

## 背景問題

- 先前版本將「孤立非零」抑制放在掃描後處理，群集計數卻在掃描中完成 → 容易出現先計數、後抹除，導致群集數暴增。
- 以 p-value 為主的判定對大樣本過於敏感，微小位移也會被認定為顯著，造成破碎的非零序列。
- 未對「正負對」高頻反向抖動做即時抑制，造成入群/出群頻繁開關。

## 設計目標

- 保持高靈敏度：在有效群集中，1 px 位移也要被視為有效。
- 入群判定延遲 1 幀：必須看到下一幀才確認入群，避免孤立非零。
- 前移抖動處理：將「正負對取消」與小幅反向抹除放入掃描主迴圈（而非事後）。
- 邊界保守：在入群邊界遇到相反號時，額外觀察第 3 幀一次，避免錯失真正起點。
- 出群穩健：連續 3 幀有效 0 才出群。

## 名詞定義

- Δpx：以像素為單位的單幀垂直位移估計（沿用現有配對/群集/統計流程後的結果）。
- candidate：本幀被視為「候選運動訊號」的條件為：
  - 通過 t-test（維持現行標準，並建議設最低樣本數）；
  - |Δpx| ≥ effect_min_px（像素域效果量門檻）；
  - 未被 camera pan 排除。
- orientation：群集方向（由入群確認時確立）。

## 狀態機

三態：Idle → PendingEnter → InCluster

1) Idle（未入群）
- 若出現 candidate：記錄 `pending_delta`、`pending_idx`，轉入 PendingEnter。
- 否則輸出 0。

2) PendingEnter（一幀延遲確認）
- 觀察下一幀：
  - 若下一幀同號或為 0 → 確認入群，以 pending 幀為起點（在此時群集計數 +1，標記 pre 圖片），轉 InCluster。
  - 若下一幀相反號：
    - 若 |d1 + d2| ≤ pair_tolerance_px → 視為典型正負對，兩幀都抹 0，回 Idle。
    - 否則 → 進入 BoundaryResolve（只在入群邊界多看第 3 幀一次）。

BoundaryResolve（入群邊界裁決）
- 觀察第 3 幀後立即決定：
  - 若第 3 幀與第 1 幀同號 → 以第 1 幀為真正起點；第 2 幀抹 0；轉 InCluster。
  - 若第 3 幀與第 2 幀同號 → 以第 2 幀為真正起點；第 1 幀抹 0；轉 InCluster。
  - 若第 3 幀為 0 或不明顯 → 保留第 1 幀為起點；第 2 幀抹 0；轉 InCluster。

3) InCluster（群集中）
- 維持 orientation（入群確立時的符號）。
- 本幀處理：
  - 若 candidate 且同號 → 直接保留。
  - 若 candidate 但相反號：
    - 若 |Δpx| ≤ jitter_max_px → 視為群內小幅抖動：
      - 優先與最近保留的一個小幅同號樣本嘗試成對抵銷，若 |p + Δpx| ≤ pair_tolerance_px，則兩者同時置 0（對齊「+1 -1 +1 -1 → 0 0 0 0」與「+1 +2 +1 -1 +2 +2 → +1 +2 0 0 +2 +2」）；
      - 否則僅將當前反向幀置 0（形成「+1 +2 +1 0 +2 +2」）。
    - 若 |Δpx| > jitter_max_px → 視為可能反轉：若連續 reversal_persist_R 幀皆為明顯反向，則關閉當前群集（標 post），並以第一個反向幀作為新 pending 重新評估下一群。
  - 若非 candidate（有效 0）→ `zero_streak += 1`，連續 `exit_zero_len`（3）幀則出群並標 post；否則仍在群中。

## JPEG 匯出與快取策略

- frame_cache：保留最近 N 幀（建議 N=20），內容為 `(frame_idx, frame)`。
- 固定深度：採用 20 幀，不做成可調；長群集不依賴快取深度。
- 入群（PendingEnter → InCluster 確認當下）：
  - pre：以「入群起點幀的前一幀」作為前 0 點；因為入群有一幀延遲，通常可由 `frame_cache[-2]` 取得；若無則退而求其次使用 `frame_cache[-1]`。
  - 執行 `export_frame_jpg((pre_idx, pre_frame), f"pre_cluster_{id:03d}.jpg", video_name)`。
- 出群（連續零達標或反向持續達標）：
  - post：以「宣告出群當下的幀」作為後 0 點。
  - 執行 `export_frame_jpg((frame_idx, frame), f"post_cluster_{id:03d}.jpg", video_name)`。
- 暗房快照：維護 `last_non_darkroom_frame` 與 `last_non_darkroom_idx`，每當處於非暗房即更新；作為強制暗房出群時的 post 來源。
- 掃描結束：若仍在群中，使用最後一幀作 post；若仍在 PendingEnter，依偏好可視為有效入群並補齊 pre/post，或放棄該 pending。
- 檔案命名維持：`pre_cluster_{id:03d}.jpg`、`post_cluster_{id:03d}.jpg`；並在 CSV `frame_path` 欄位對應回填。

## 參數與預設

- enter_delay_frames: 1（固定，一幀延遲確認入群）。
- pair_tolerance_px: 1.0（成對相加容忍值）。
- jitter_max_px: 1.0（群內視為抖動的反向幅度上限）。
- exit_zero_len: 3（出群所需連續有效 0 幀數）。
- reversal_persist_R: 2（連續幀的明顯反向才視為真反轉）。
- effect_min_px: 1.0（像素域效果量門檻；你要求 1px 也有效 → 以像素定義穩定）。
- t-test 顯著門檻：沿用現行；建議加入最低樣本數 `min_matches = 6`。

上述參數以像素域為主，避免比例尺變動造成行為不一致；最終 mm 累加在 CSV 與疊加顯示仍照現有比例尺換算。

## 輸出與重現性（最終決策）

- CSV 最小欄位：`frame_idx`, `second`, `vertical_travel_distance (mm)`, `frame_path`, `camera_pan`, `cluster_id`, `orientation`, `darkroom_event`（其值為 `enter_darkroom` / `exit_darkroom` / 空字串）。
- Debug 模式（`LIFT_DEBUG_STATE=1`）時才額外輸出：`state`, `delta_px`, `is_candidate`, `matches_count`。
- inspection MP4：不顯示 enter/exit 字樣；顯示 `cluster_id` 與 `orientation`；被抹除幀以 `jitter` 呈現（沿用 camera pan 類似風格）；暗房仍顯示 `darkroom (ignored)`。
- KMeans 可重現性：設定 `KMeans(n_clusters=2, random_state=0)` 以固定隨機種子，避免分群初始化帶來的細微差異。

## 與現有程式的整合點

- 保留：ORB 特徵、BFMatcher、KMeans 兩群、水平位移 t-test 判斷 camera pan、垂直方向以兩群中位數差為估計的既有流程。
- 調整點：
  - 在取得 Δpx 後，先進入「狀態機 + 抖動處理」再決定是否承認非零（並觸發入群/出群與 pre/post 匯出）。
  - 群集計數只在「入群確認」時 +1，避免先數後抹。
  - `frame_path` 的 pre/post 命名與回填改為在該時機執行（靠 `frame_cache` 與 pending）。
  - 結束收尾：若群未關閉，補 post；若仍有 pending，依策略處理。
  - 分群：`KMeans(n_clusters=2, random_state=0)` 強化可重現性。

## 不變量與風險控制（收斂檢查）

- 影片處理完成後檢查：
  - `nonzero_count = sum(abs(v_mm) > 0 for v_mm in result['v_travel_distance'])`
  - `cluster_count = physical_cluster_counter`
  - 期望 `cluster_count ≤ nonzero_count`；若違反，列印警告並輸出診斷資訊（參數、樣本數、邊界決策日誌）。

## 邏輯對照（關鍵案例）

- 0, +2, -3, 0 → `+2` 與 `-3` 相加 ≈ -1，|sum| ≤ 1 → 兩幀抹 0。
- +1, -1, +1, -1 → 逐對 |sum| = 0 → 全抹 0。
- -1, +2, -1, -2, -3（入群邊界）→ 經 BoundaryResolve，保留第一個 -1 為起點，+2 抹 0。
- 群內弱抖動 +3, +3, +4, +2, +3 → 全保留，不做調整。
- 群內小反向 +1, +2, +1, -1, +2, +2 → 優先最小成對抵銷 → +1, +2, 0, 0, +2, +2；若無合適成對，僅抹反向 → +1, +2, +1, 0, +2, +2。

## 實作步驟（待辦）

1) 在 `scan()` 主迴圈中整合三態狀態機與一幀延遲入群判定。
2) 實作群內「最小正負對優先抵銷」與小幅反向抹除；加入 `reversal_persist_R` 的真反轉判斷。
3) 調整 pre/post JPEG 匯出與 `frame_path` 回填時機；完善 `frame_cache` 的使用與結束收尾邏輯。
4) 增加不變量檢查與詳細警告輸出（參數、樣本數、邊界決策）。
5) 參數化（環境變數或 Config）：`pair_tolerance_px`、`jitter_max_px`、`exit_zero_len`、`reversal_persist_R`、`effect_min_px`、`min_matches`。

## 風險與緩解

- 邊界裁決錯誤導致起點偏移：BoundaryResolve 只在入群邊界多看 1 幀，並偏好保留第 1 幀為起點；同時輸出決策日誌便於追溯。
- 比例尺異常（0/NaN/極小）：檢測 `scale_factor` 非法時回退為 1.0 並加強警告，避免數值放大。
- 效果量與 t-test 衝突：以像素門檻作為最低效果量，t-test 僅輔助；可透過參數調整靈敏度。

## 暗房最終規則（統整）

- 進入暗房時：
  - 若在 PendingEnter：一律清空 pending，不入群。
  - 若在 InCluster：一律強制出群；post 使用 `last_non_darkroom_frame`（若無可用快照，退回最近可用幀或當前幀）。
- 暗房期間：不入群、結果輸出為 0；CSV 只在切換點標 `darkroom_event`（enter/exit），不輸出 per-frame `in_darkroom` 布林欄。

——

本計畫完成後，應能穩定保證「群集數 ≤ 非零點數」，同時保留 1px 級別的靈敏偵測，並避免高頻反向抖動造成的群集暴增。


