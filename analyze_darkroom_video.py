"""
暗房影片分析腳本 - 評估自動化偵測可行性
"""
import cv2
import numpy as np
import os
import sys

# 確保在專案根目錄執行
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append("src/")

VIDEO_PATH = "lifts/darkroom_data/21_a.mp4"
DARKROOM_START = 30  # 秒
DARKROOM_END = 508   # 8:28 = 508秒
MOTION_TIME = 185    # 3:05 = 185秒

# 關鍵時間點分析
ANALYSIS_TIMESTAMPS = [
    30,    # 暗房開始
    60,    # 暗房早期
    185,   # 有運動發生
    186,   # 運動後一秒
    300,   # 暗房中期
    500,   # 暗房結束前（亮度不穩定）
]

def extract_frame(video_path, timestamp_seconds):
    """精確提取指定秒數的幀"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame, frame_number, fps
    return None, None, None

def analyze_frame_quality(frame, label):
    """分析幀的畫質特徵"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 基本統計
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    min_val, max_val = np.min(gray), np.max(gray)

    # 邊緣強度
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size

    # 梯度強度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)

    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  亮度統計:")
    print(f"    平均亮度: {mean_brightness:.2f} (0-255)")
    print(f"    標準差:   {std_brightness:.2f}")
    print(f"    範圍:     {min_val} - {max_val}")
    print(f"  邊緣特徵:")
    print(f"    邊緣密度: {edge_density:.4f} ({edge_density*100:.2f}%)")
    print(f"    梯度強度: {mean_gradient:.2f}")

    return {
        'mean_brightness': mean_brightness,
        'edge_density': edge_density,
        'mean_gradient': mean_gradient,
        'gray': gray
    }

def test_feature_detectors(frame, label):
    """測試不同特徵點偵測器的效果"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(f"\n  🔍 特徵點偵測測試:")

    # 1. ORB (原始方法)
    orb = cv2.ORB.create(nfeatures=100)
    kp_orb, _ = orb.detectAndCompute(gray, None)
    print(f"    ORB (原始):     {len(kp_orb)} 個特徵點")

    # 2. ORB + CLAHE (對比度增強)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    kp_orb_clahe, _ = orb.detectAndCompute(enhanced, None)
    print(f"    ORB + CLAHE:    {len(kp_orb_clahe)} 個特徵點")

    # 3. ORB + Canny 邊緣遮罩
    edges = cv2.Canny(gray, 30, 100)
    edges_dilated = cv2.dilate(edges, np.ones((5,5), np.uint8))
    kp_orb_edge, _ = orb.detectAndCompute(gray, edges_dilated)
    print(f"    ORB + Canny遮罩: {len(kp_orb_edge)} 個特徵點")

    # 4. SIFT (更強大但較慢)
    try:
        sift = cv2.SIFT_create(nfeatures=100)
        kp_sift, _ = sift.detectAndCompute(gray, None)
        print(f"    SIFT (原始):    {len(kp_sift)} 個特徵點")

        kp_sift_clahe, _ = sift.detectAndCompute(enhanced, None)
        print(f"    SIFT + CLAHE:   {len(kp_sift_clahe)} 個特徵點")
    except:
        print(f"    SIFT: 不可用 (可能需要 opencv-contrib-python)")

    return {
        'orb': len(kp_orb),
        'orb_clahe': len(kp_orb_clahe),
        'orb_edge': len(kp_orb_edge)
    }

def save_comparison_images(frame, timestamp, output_dir="lifts/darkroom_analysis"):
    """儲存不同處理方法的對比圖"""
    os.makedirs(output_dir, exist_ok=True)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # CLAHE 增強
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Canny 邊緣
    edges = cv2.Canny(gray, 30, 100)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # ORB 特徵點 (原始)
    orb = cv2.ORB.create(nfeatures=100)
    kp_orb, _ = orb.detectAndCompute(gray, None)
    frame_orb = frame.copy()
    cv2.drawKeypoints(frame_orb, kp_orb, frame_orb, color=(0, 255, 0), flags=0)

    # ORB 特徵點 (CLAHE)
    kp_orb_clahe, _ = orb.detectAndCompute(enhanced, None)
    frame_orb_clahe = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.drawKeypoints(frame_orb_clahe, kp_orb_clahe, frame_orb_clahe, color=(0, 255, 0), flags=0)

    # 組合成 2x3 網格
    row1 = np.hstack([frame, cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), edges_color])
    row2 = np.hstack([frame_orb, frame_orb_clahe, np.zeros_like(frame)])
    combined = np.vstack([row1, row2])

    # 添加標籤
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "CLAHE Enhanced", (w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, "Canny Edges", (2*w+10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"ORB ({len(kp_orb)} pts)", (10, h+30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, f"ORB+CLAHE ({len(kp_orb_clahe)} pts)", (w+10, h+30), font, 1, (255, 255, 255), 2)

    output_path = os.path.join(output_dir, f"analysis_{int(timestamp)}s.jpg")
    cv2.imwrite(output_path, combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"  💾 已儲存分析圖: {output_path}")

def main():
    print("🌙 暗房影片可行性分析")
    print("="*60)

    # 檢查影片
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 影片不存在: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"📹 影片資訊:")
    print(f"  路徑: {VIDEO_PATH}")
    print(f"  FPS: {fps:.2f}")
    print(f"  總長度: {duration:.1f}秒 ({int(duration//60)}分{int(duration%60)}秒)")
    print(f"  暗房區間: {DARKROOM_START}s - {DARKROOM_END}s ({(DARKROOM_END-DARKROOM_START)/60:.1f}分鐘)")

    # 分析關鍵時間點
    results = []
    for timestamp in ANALYSIS_TIMESTAMPS:
        frame, frame_num, _ = extract_frame(VIDEO_PATH, timestamp)
        if frame is None:
            print(f"❌ 無法提取 {timestamp}s 的幀")
            continue

        label = f"時間 {timestamp}s (第 {frame_num} 幀)"
        if timestamp == MOTION_TIME:
            label += " 🔄 [運動點]"
        elif timestamp >= 500:
            label += " ⚠️ [亮度不穩定]"

        quality = analyze_frame_quality(frame, label)
        features = test_feature_detectors(frame, label)
        save_comparison_images(frame, timestamp)

        results.append({
            'timestamp': timestamp,
            'quality': quality,
            'features': features
        })

    # 總結報告
    print(f"\n{'='*60}")
    print("📊 分析總結")
    print(f"{'='*60}")

    avg_orb = np.mean([r['features']['orb'] for r in results])
    avg_orb_clahe = np.mean([r['features']['orb_clahe'] for r in results])
    avg_orb_edge = np.mean([r['features']['orb_edge'] for r in results])

    print(f"\n特徵點數量統計 (平均):")
    print(f"  ORB 原始:        {avg_orb:.1f} 個")
    print(f"  ORB + CLAHE:     {avg_orb_clahe:.1f} 個")
    print(f"  ORB + Canny遮罩: {avg_orb_edge:.1f} 個")

    print(f"\n建議:")
    if avg_orb_clahe >= 20:
        print("  ✅ CLAHE增強後特徵點充足，建議採用做法A (自動化偵測)")
        print("  ✅ 建議流程: CLAHE增強 → ORB特徵點 → 原有匹配與分群算法")
    elif avg_orb_edge >= 15:
        print("  ⚠️  需要結合Canny邊緣遮罩，做法A可行但需要更多調適")
        print("  ⚠️  建議流程: Canny邊緣 → 膨脹遮罩 → ORB特徵點 → 匹配與分群")
    else:
        print("  ❌ 特徵點不足，建議採用做法B (人工輔助偵測)")
        print("  ❌ 或考慮混合方案: 初步自動偵測 + 人工校正補充")

    print(f"\n分析圖已儲存至: lifts/darkroom_analysis/")
    print("請檢視圖片以評估視覺效果")

if __name__ == "__main__":
    main()