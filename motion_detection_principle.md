### 繁體中文版：影片運動分析原理

本程式旨在分析影片，以量化顯微鏡鏡組的垂直運動。其原理結合了特徵追蹤、統計分群與物理單位校準。

#### 1. 尋找並追蹤特徵點
程式首先在影像中識別出穩定、易於辨識的「特徵點」（如紋理邊角）。接著，它會追蹤這些點在連續畫面間的移動軌跡，從而得到一系列代表運動路徑的「運動向量」。

#### 2. 分群以區分運動與背景
得到的運動向量包含兩種：來自靜止背景的（可能源於攝影機晃動）和來自移動中鏡組的。我們使用 `K-Means` 統計分群法，將所有運動向量基於其移動距離，自動分為兩群：移動接近零的「背景群」和有顯著移動的「運動群」。

#### 3. 檢定並計算真實位移
在確認位移前，程式會對每一群的運動數據進行 `t-test` 統計檢定。此步驟的目的是判斷觀測到的移動是否具有統計顯著性，而非隨機雜訊。只有通過檢定的運動，才被視為有效。

鏡組的真實垂直位移，是「運動群」與「背景群」各自的顯著移動距離之差。這種差分計算能有效排除攝影機抖動等全域性運動的干擾，分離出鏡組的純粹運動。

#### 4. 透過比例尺轉換為物理單位
前述計算出的位移是像素（pixel）單位。為了得到有物理意義的毫米（mm）值，程式會使用一個「比例尺」進行換算。這個比例尺是透過分析數張校準圖片預先計算而得的：操作員在這些圖片上手動標記一段已知長度（例如 10mm）的兩個端點，程式則測量這兩點間的像素距離，從而建立像素與毫米之間的精確轉換比例。

最終，經過單位換算的位移量會連同時間點被記錄至 CSV 檔案中。

---

### English Version: Methodology

This method quantifies the vertical motion of a microscope lens group from video sequences by integrating feature-point tracking, unsupervised clustering, and scale calibration.

**1. Feature Tracking:**
The ORB (Oriented FAST and Rotated BRIEF) algorithm detects and computes descriptors for keypoints within a region of interest. A Brute-Force matcher then establishes correspondences between consecutive frames, yielding displacement vectors for stable features.

**2. Motion Segmentation via Clustering:**
To distinguish object motion from background noise (e.g., camera jitter), K-Means clustering (`n_clusters=2`) partitions the displacement vectors into a "static" cluster (background) and a "dynamic" cluster (moving lens group) based on their travel distance.

**3. Displacement Validation and Quantification:**
A one-sample t-test is applied to each cluster to validate that its displacement is statistically significant against a null hypothesis of zero motion. The object's true vertical displacement is then calculated as the relative difference between the significant median vertical displacements of the dynamic and static clusters. This differential measurement isolates the lens group's motion from global camera movement.

**4. Scale Calibration for Physical Units:**
To convert pixel displacement into physical units (mm), a scale factor is pre-calibrated. This is achieved by analyzing reference images where a known physical distance (e.g., 10mm) has been manually annotated with markers. The system measures the pixel distance between these markers to establish a precise pixels-per-millimeter ratio, which is then used for the final unit conversion.

The resulting displacement data is logged to a time-series CSV file.