"""
測試短片段 (21a_tinyclip.mp4) - 使用 Canny Edge Detection 前處理
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from rotation_utils import rotate_frame

# === Configuration ===
VIDEO_PATH = r"D:\Lift_project\lifts\test_short\21a_tinyclip.mp4"
ROTATION_ANGLE = 21.0  # From rotation_config
ROI_RATIO = 0.6
FRAME_INTERVAL = 6
OUTPUT_DIR = r"D:\Lift_project\lifts\test_short\results_canny"

# Canny parameters
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150
CANNY_BLUR_SIZE = 5  # GaussianBlur before Canny

# Motion thresholds
MOTION_MIN_MATCHES = 8
MOTION_MIN_DISPLACEMENT = 25

# === Functions ===
def preprocess_canny(frame):
    """Canny Edge Detection 前處理"""
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Optional: Gaussian blur to reduce noise
    if CANNY_BLUR_SIZE > 0:
        gray = cv2.GaussianBlur(gray, (CANNY_BLUR_SIZE, CANNY_BLUR_SIZE), 0)

    # Canny edge detection
    edges = cv2.Canny(gray, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    return edges

def create_roi_mask(shape, ratio):
    """建立 ROI 遮罩"""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    roi_y1 = int(h * (1 - ratio) / 2)
    roi_y2 = int(h * (1 + ratio) / 2)
    roi_x1 = int(w * (1 - ratio) / 2)
    roi_x2 = int(w * (1 + ratio) / 2)

    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
    return mask, (roi_x1, roi_y1, roi_x2, roi_y2)

def draw_matches_and_motion(frame, kp, matches, vertical_displacements, roi_bounds, result_type, edge_frame=None):
    """繪製特徵點、匹配線、ROI 框 (並顯示邊緣圖)"""
    vis = frame.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds

    # Draw ROI
    cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)

    # Draw keypoints
    for pt in kp:
        x, y = int(pt.pt[0]), int(pt.pt[1])
        cv2.circle(vis, (x, y), 3, (0, 255, 0), 1)

    # Draw matches and vertical lines
    if len(matches) > 0:
        for i, (p1, p2) in enumerate(matches):
            x1, y1 = int(p1[0]), int(p1[1])
            x2, y2 = int(p2[0]), int(p2[1])

            # Match line (red)
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Vertical displacement line (green)
            cv2.line(vis, (x1, y1), (x1, y2), (0, 255, 0), 1)

    # Add text info
    v_travel = int(np.median(vertical_displacements)) if len(vertical_displacements) > 0 else 0
    info_text = f"V-Travel: {v_travel} px | Matches: {len(matches)} | {result_type}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    # Optional: Show edge detection result in corner
    if edge_frame is not None:
        h, w = vis.shape[:2]
        edge_h, edge_w = edge_frame.shape[:2]

        # Resize edge to small corner (1/4 size)
        small_h = h // 4
        small_w = w // 4
        edge_small = cv2.resize(edge_frame, (small_w, small_h))

        # Convert to BGR for overlay
        edge_bgr = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2BGR)

        # Overlay in top-right corner
        vis[0:small_h, w-small_w:w] = edge_bgr

    return vis

# === Main ===
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {VIDEO_PATH}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f}s")
    print(f"Processing every {FRAME_INTERVAL} frames")
    print(f"Canny parameters: T1={CANNY_THRESHOLD1}, T2={CANNY_THRESHOLD2}, Blur={CANNY_BLUR_SIZE}")
    print()

    # ORB detector
    orb = cv2.ORB_create(nfeatures=100)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Sequential reading from start (NO cap.set!)
    frame_idx = 0
    prev_frame = None
    prev_kp = None
    prev_desc = None
    prev_edges = None

    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Process every FRAME_INTERVAL frames
        if frame_idx % FRAME_INTERVAL != 0:
            continue

        timestamp = frame_idx / fps

        # 1. Rotate frame
        frame_rotated = rotate_frame(frame, ROTATION_ANGLE)

        # 2. Create ROI mask (on rotated frame)
        mask, roi_bounds = create_roi_mask(frame_rotated.shape, ROI_RATIO)

        # 3. Canny edge detection
        frame_edges = preprocess_canny(frame_rotated)

        # 4. Feature detection on edge image
        kp, desc = orb.detectAndCompute(frame_edges, mask)

        if prev_desc is not None and desc is not None:
            # Match features
            matches = bf.knnMatch(prev_desc, desc, k=2)

            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= MOTION_MIN_MATCHES:
                # Calculate vertical displacements
                vertical_displacements = []
                match_pairs = []

                for m in good_matches:
                    pt1 = prev_kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    dy = pt2[1] - pt1[1]
                    vertical_displacements.append(dy)
                    match_pairs.append((pt1, pt2))

                # Calculate median
                median_dy = np.median(vertical_displacements)

                # Determine motion
                if abs(median_dy) >= MOTION_MIN_DISPLACEMENT:
                    result_type = "motion"
                    direction = "down" if median_dy > 0 else "up"
                    print(f"Frame {frame_idx:4d} ({timestamp:6.2f}s): MOTION - {int(median_dy):3d} px {direction} | Matches: {len(good_matches)}")
                else:
                    result_type = "static"
                    print(f"Frame {frame_idx:4d} ({timestamp:6.2f}s): static - {int(median_dy):3d} px | Matches: {len(good_matches)}")

                # Draw visualization (with edge detection preview)
                vis = draw_matches_and_motion(frame_rotated, kp, match_pairs,
                                             vertical_displacements, roi_bounds,
                                             result_type, frame_edges)

                # Save image
                output_path = os.path.join(OUTPUT_DIR,
                    f"frame_{frame_idx:05d}_{timestamp:.1f}s_{result_type}_canny.jpg")
                cv2.imwrite(output_path, vis)

                results.append({
                    'frame': frame_idx,
                    'time': timestamp,
                    'matches': len(good_matches),
                    'displacement': int(median_dy),
                    'type': result_type
                })
            else:
                print(f"Frame {frame_idx:4d} ({timestamp:6.2f}s): insufficient matches ({len(good_matches)})")
        else:
            print(f"Frame {frame_idx:4d} ({timestamp:6.2f}s): first frame or no descriptors")

        # Update previous frame
        prev_frame = frame_rotated
        prev_kp = kp
        prev_desc = desc
        prev_edges = frame_edges

    cap.release()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    motion_frames = [r for r in results if r['type'] == 'motion']
    static_frames = [r for r in results if r['type'] == 'static']

    print(f"Total processed: {len(results)} frame pairs")
    print(f"Motion detected: {len(motion_frames)} frames")
    print(f"Static: {len(static_frames)} frames")

    if motion_frames:
        print("\nMotion frames:")
        for r in motion_frames:
            print(f"  Frame {r['frame']:4d} ({r['time']:6.2f}s): {r['displacement']:3d} px | {r['matches']} matches")

    print(f"\nOutput saved to: {OUTPUT_DIR}")