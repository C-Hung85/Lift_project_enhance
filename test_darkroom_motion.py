"""
階段 1: 暗房運動偵測驗證腳本

目標: 測試完整的運動偵測 pipeline 在暗房區間的可行性
測試範圍: 21_a.mp4 的 3:00-3:15 (包含運動點 3:05-3:07)
"""
import os
import sys
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("src/")

import cv2
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans

# 測試參數
VIDEO_PATH = "lifts/darkroom_data/21_a.mp4"
TEST_START = 180      # 3:00 (秒)
TEST_END = 195        # 3:15 (秒)
MOTION_POINT = 185    # 3:05 (預期運動點)
FRAME_INTERVAL = 6    # 與主程式一致
ROI_RATIO = 0.6       # 與主程式一致
MIN_MATCHES = 6       # 最低匹配對數

# 輸出目錄
OUTPUT_DIR = "lifts/darkroom_motion_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_darkroom_frame(frame):
    """
    暗房影片專用前處理 - CLAHE 增強

    Args:
        frame: BGR 彩色影像

    Returns:
        enhanced: CLAHE 增強後的灰階影像
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def remove_outliers(data, method='iqr'):
    """移除離群值（與主程式邏輯一致）"""
    if len(data) < 3:
        return np.arange(len(data))

    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (data >= lower_bound) & (data <= upper_bound)
    return np.where(mask)[0]

def test_motion_detection():
    """執行運動偵測測試"""

    print("🌙 暗房運動偵測驗證測試")
    print("="*70)

    # 開啟影片
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 影片資訊:")
    print(f"  FPS: {fps:.2f}")
    print(f"  解析度: {width}x{height}")
    print(f"  測試區間: {TEST_START}s - {TEST_END}s ({TEST_END-TEST_START}秒)")
    print(f"  幀間隔: 每 {FRAME_INTERVAL} 幀取樣")

    # 建立 ROI 遮罩
    mask = np.zeros((height, width), dtype=np.uint8)
    roi_y1 = int(height * ROI_RATIO / 2)
    roi_y2 = int(height * (1 - ROI_RATIO / 2))
    roi_x1 = int(width * ROI_RATIO / 2)
    roi_x2 = int(width * (1 - ROI_RATIO / 2))
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1

    print(f"  ROI 範圍: ({roi_x1}, {roi_y1}) - ({roi_x2}, {roi_y2})")

    # 初始化偵測器
    orb = cv2.ORB.create(nfeatures=100)
    bf_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    kmeans = KMeans(n_clusters=2, random_state=0)

    # 跳到測試起點
    start_frame = int(TEST_START * fps)
    end_frame = int(TEST_END * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 讀取第一幀
    ret, frame1 = cap.read()
    if not ret:
        print("❌ 無法讀取起始幀")
        return

    # 前處理 + 特徵偵測
    enhanced1 = preprocess_darkroom_frame(frame1)
    kp1, desc1 = orb.detectAndCompute(enhanced1, mask)

    print(f"\n✅ 初始幀特徵點: {len(kp1)} 個")

    # 儲存第一幀的視覺化
    vis_frame1 = frame1.copy()
    cv2.drawKeypoints(vis_frame1, kp1, vis_frame1, color=(0, 255, 0), flags=0)
    cv2.rectangle(vis_frame1, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "frame1_features.jpg"), vis_frame1)

    # 測試結果容器
    results = []

    print(f"\n{'='*70}")
    print(f"{'Frame':>6} {'Time':>7} {'KP1':>5} {'KP2':>5} {'Matches':>8} {'V-Travel':>10} {'P-value':>10} {'Status':>10}")
    print(f"{'='*70}")

    frame_idx = start_frame

    while True:
        # 讀取下一幀
        for _ in range(FRAME_INTERVAL):
            ret, frame2 = cap.read()
            frame_idx += 1

        if not ret or frame_idx >= end_frame:
            break

        time_sec = frame_idx / fps

        # 前處理 + 特徵偵測
        enhanced2 = preprocess_darkroom_frame(frame2)
        kp2, desc2 = orb.detectAndCompute(enhanced2, mask)

        # 初始化結果
        vertical_travel = 0
        camera_pan = False
        num_matches = 0
        pvalue = 1.0
        status = "no_feature"

        # 特徵匹配
        if desc1 is not None and desc2 is not None and len(kp1) > 0 and len(kp2) > 0:
            matches = bf_matcher.match(desc2, desc1)
            num_matches = len(matches)

            if num_matches >= MIN_MATCHES:
                # 建立配對陣列
                paired_info = []
                for m in matches:
                    kp1_coord = np.array(kp1[m.trainIdx].pt)
                    kp2_coord = np.array(kp2[m.queryIdx].pt)

                    distance = np.linalg.norm(kp1_coord - kp2_coord)
                    h_travel = kp2_coord[0] - kp1_coord[0]
                    v_travel = kp2_coord[1] - kp1_coord[1]

                    paired_info.append([m.queryIdx, m.trainIdx, distance, h_travel, v_travel])

                paired_info = np.array(paired_info)

                # 移除離群值
                valid_idx = remove_outliers(paired_info[:, 2])
                paired_info = paired_info[valid_idx]

                if len(paired_info) >= MIN_MATCHES:
                    # 檢查 camera pan
                    h_travels = paired_info[:, 3]
                    pvalue = ttest_1samp(h_travels, 0).pvalue
                    camera_pan = pvalue < 0.001

                    if not camera_pan:
                        # K-means 分群
                        distances = paired_info[:, 2].reshape(-1, 1)
                        if len(set(distances.flatten())) > 1:
                            labels = kmeans.fit_predict(distances)

                            # 兩群的垂直位移
                            group0_v = paired_info[labels == 0, 4]
                            group1_v = paired_info[labels == 1, 4]

                            pvalue0 = ttest_1samp(group0_v, 0).pvalue
                            pvalue1 = ttest_1samp(group1_v, 0).pvalue

                            v0 = np.median(group0_v) if pvalue0 < 0.0001 else 0
                            v1 = np.median(group1_v) if pvalue1 < 0.0001 else 0

                            if abs(v0) > abs(v1):
                                vertical_travel = int(v1 - v0)
                            else:
                                vertical_travel = int(v0 - v1)

                            status = "motion" if vertical_travel != 0 else "static"
                            pvalue = min(pvalue0, pvalue1)
                        else:
                            status = "no_cluster"
                    else:
                        status = "camera_pan"
                else:
                    status = "few_matches"
            else:
                status = "few_matches"

        # 記錄結果
        results.append({
            'frame_idx': frame_idx,
            'time': time_sec,
            'kp1': len(kp1),
            'kp2': len(kp2),
            'matches': num_matches,
            'vertical_travel': vertical_travel,
            'pvalue': pvalue,
            'camera_pan': camera_pan,
            'status': status
        })

        # 輸出日誌
        status_icon = "✅" if status == "motion" else ("⚠️" if status == "camera_pan" else "")
        print(f"{frame_idx:6d} {time_sec:7.2f} {len(kp1):5d} {len(kp2):5d} {num_matches:8d} {vertical_travel:10d} {pvalue:10.4f} {status_icon}{status:>9}")

        # 儲存關鍵幀的視覺化
        if abs(time_sec - MOTION_POINT) < 3:  # 運動點附近 ±3 秒
            vis_frame = frame2.copy()

            # 繪製特徵點
            cv2.drawKeypoints(vis_frame, kp2, vis_frame, color=(0, 255, 0), flags=0)

            # 繪製匹配線
            if desc1 is not None and desc2 is not None:
                matches = bf_matcher.match(desc2, desc1)
                for m in matches[:20]:  # 只繪製前 20 條
                    pt1 = tuple(map(int, kp1[m.trainIdx].pt))
                    pt2 = tuple(map(int, kp2[m.queryIdx].pt))
                    cv2.line(vis_frame, pt1, pt2, (0, 0, 255), 1)

            # 繪製 ROI
            cv2.rectangle(vis_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

            # 繪製資訊
            info_text = f"Frame {frame_idx} | {time_sec:.1f}s | {status}"
            cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)

            travel_text = f"V-Travel: {vertical_travel} px | Matches: {num_matches}"
            cv2.putText(vis_frame, travel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2)

            filename = f"frame_{frame_idx}_{time_sec:.1f}s_{status}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, filename), vis_frame)

        # 更新前一幀
        kp1, desc1 = kp2, desc2

    cap.release()

    # 統計分析
    print(f"\n{'='*70}")
    print("📊 測試結果統計")
    print(f"{'='*70}")

    total_frames = len(results)
    motion_frames = sum(1 for r in results if r['status'] == 'motion')
    static_frames = sum(1 for r in results if r['status'] == 'static')
    pan_frames = sum(1 for r in results if r['camera_pan'])
    failed_frames = sum(1 for r in results if r['status'] in ['no_feature', 'few_matches', 'no_cluster'])

    print(f"\n總幀數: {total_frames}")
    print(f"  ✅ 偵測到運動: {motion_frames} 幀 ({motion_frames/total_frames*100:.1f}%)")
    print(f"  ⚪ 靜止狀態:   {static_frames} 幀 ({static_frames/total_frames*100:.1f}%)")
    print(f"  ⚠️  Camera Pan: {pan_frames} 幀 ({pan_frames/total_frames*100:.1f}%)")
    print(f"  ❌ 偵測失敗:   {failed_frames} 幀 ({failed_frames/total_frames*100:.1f}%)")

    # 特徵點統計
    avg_kp = np.mean([r['kp2'] for r in results])
    avg_matches = np.mean([r['matches'] for r in results])

    print(f"\n特徵點統計:")
    print(f"  平均特徵點數: {avg_kp:.1f} 個")
    print(f"  平均匹配對數: {avg_matches:.1f} 對")

    # 運動統計
    motion_results = [r for r in results if r['status'] == 'motion']
    if motion_results:
        total_travel = sum(abs(r['vertical_travel']) for r in motion_results)
        print(f"\n運動統計:")
        print(f"  偵測到運動的幀數: {len(motion_results)}")
        print(f"  總垂直位移: {total_travel} 像素")
        print(f"  平均位移/幀: {total_travel/len(motion_results):.2f} 像素")

    # 成功率評估
    success_rate = (motion_frames + static_frames) / total_frames * 100

    print(f"\n{'='*70}")
    print("🎯 可行性評估")
    print(f"{'='*70}")
    print(f"\n偵測成功率: {success_rate:.1f}%")

    if success_rate >= 70:
        print("\n✅ 評估結果: 做法 A（自動化偵測）完全可行")
        print("   建議: 進入階段 2，開發完整的暗房專用主程式")
    elif success_rate >= 50:
        print("\n⚠️  評估結果: 做法 A 可行但需要調整")
        print("   建議: 調整參數後重新測試，或考慮混合方案")
    else:
        print("\n❌ 評估結果: 做法 A 不可行")
        print("   建議: 採用做法 B（人工輔助偵測）或混合方案")

    print(f"\n視覺化結果已儲存至: {OUTPUT_DIR}/")

    # 儲存詳細報告
    report_path = os.path.join(OUTPUT_DIR, "test_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("暗房運動偵測驗證報告\n")
        f.write("="*70 + "\n\n")
        f.write(f"測試影片: {VIDEO_PATH}\n")
        f.write(f"測試區間: {TEST_START}s - {TEST_END}s\n")
        f.write(f"總幀數: {total_frames}\n\n")
        f.write("詳細結果:\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"Frame {r['frame_idx']:6d} | {r['time']:7.2f}s | "
                   f"KP: {r['kp2']:3d} | Matches: {r['matches']:3d} | "
                   f"V-Travel: {r['vertical_travel']:5d} px | "
                   f"Status: {r['status']}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"成功率: {success_rate:.1f}%\n")

    print(f"詳細報告已儲存至: {report_path}")

if __name__ == "__main__":
    test_motion_detection()