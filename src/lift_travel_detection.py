import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import warnings
import utils
import datetime
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans
from multiprocessing import Pool
from config import Config, video_config
try:
    from rotation_config import rotation_config
except ImportError:
    # 如果 rotation_config.py 不存在，使用空字典
    rotation_config = {}
from rotation_utils import rotate_frame

try:
    from darkroom_intervals import darkroom_intervals
except ImportError:
    # 如果 darkroom_intervals.py 不存在，使用空字典
    darkroom_intervals = {}
from darkroom_utils import get_darkroom_intervals_for_video, is_in_darkroom_interval

def export_frame_jpg(frame_data, jpg_filename, video_name):
    """匯出單個幀為JPG"""
    frame_idx, frame = frame_data

    # 建立匯出目錄路徑
    export_dir = os.path.join('lifts', 'exported_frames', video_name)
    os.makedirs(export_dir, exist_ok=True)

    export_path = os.path.join(export_dir, jpg_filename)

    # 設定JPG壓縮參數（高品質）
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

    # 匯出原始未處理的幀
    success = cv2.imwrite(export_path, frame, encode_param)
    if success:
        print(f"📸 匯出JPG: {jpg_filename} (frame {frame_idx})")
    else:
        print(f"❌ 匯出失敗: {jpg_filename}")

    return success

from scale_cache_utils import (
    load_scale_cache, 
    save_scale_cache, 
    is_cache_valid, 
    get_missing_videos,
    print_cache_status,
    generate_scale_images_hash
)

warnings.filterwarnings('ignore')

# Parameters and objects
DATA_FOLDER = Config['files']['data_folder']
feature_detector = cv2.ORB.create(nfeatures=100)
feature_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING, crossCheck=True)
cluster = KMeans(n_clusters=2)
FRAME_INTERVAL = Config['scan_setting']['interval']
ROI_RATIO = 0.6

# Probe logging controls (avoid spam; configurable via env)
PROBE_MODE = os.getenv('LIFT_PROBE', '1') == '1'  # 預設開啟，設為 0 可關閉
PROBE_INTERVAL_FRAMES = int(os.getenv('LIFT_PROBE_INTERVAL', '1000'))
PROBE_FRAMES = os.getenv('LIFT_PROBE_FRAMES', '6858,6906,9000,20274,58836')
PROBE_FRAMES_SET = set(int(x) for x in PROBE_FRAMES.split(',') if x.strip().isdigit())

# 可選：鎖定僅處理單一檔名（例如 1.mp4）
TARGET_VIDEO = os.getenv('LIFT_TARGET')  # 若為 None 則處理全部

# create necessary folders
for folder_name in ['inspection', 'result']:
    os.makedirs(os.path.join(DATA_FOLDER, 'lifts', folder_name), exist_ok=True)

def scan(video_path, file_name):
    vidcap = cv2.VideoCapture(video_path)
    ret, frame = vidcap.read()
    h, w = frame.shape[:2]
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    start_frame = int(video_config.get(file_name, {}).get('start', 0) * fps)
    end_frame = int(video_config.get(file_name, {}).get('end', video_length/fps) * fps)
    roi_ratio = video_config.get(file_name, {}).get('roi_ratio', ROI_RATIO)

    # probe: 基本資訊
    if PROBE_MODE:
        print(f"[probe] video={file_name} fps={fps:.6f} total_frames={video_length} start_frame={start_frame} end_frame={end_frame}")

    # 取得暗房時間區間設定
    darkroom_intervals_seconds, has_darkroom = get_darkroom_intervals_for_video(file_name, darkroom_intervals)

    # create a mask to define the ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h*roi_ratio/2):int(h*(1-roi_ratio/2)), int(w*roi_ratio/2):int(w*(1-roi_ratio/2))] = 1

    # record container
    result = {
        'frame':[],
        'frame_idx':[],
        'keypoints':[],
        'camera_pan':[],
        'v_travel_distance':[],
        'kp_pair_lines':[],
        'frame_path':[]     # 新增：對應的匯出圖片路徑標籤
    }

    # 物理群集檢測變數
    physical_cluster_counter = 0      # 物理群集序號計數器
    in_physical_cluster = False       # 是否在物理群集中
    current_cluster_id = None         # 當前物理群集ID
    frame_cache = []                  # 緩存最近幀：[(frame_idx, frame), ...]
    pending_pre_export = None         # 待匯出的前0點幀

    # detect keypoints
    keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame, mask)

    # set video to the start point
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while ret:
        # pre-read probe
        frame_idx = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        pre_ms = vidcap.get(cv2.CAP_PROP_POS_MSEC)

        ret, frame = vidcap.read()
        post_idx = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        post_ms = vidcap.get(cv2.CAP_PROP_POS_MSEC)

        if PROBE_MODE and (frame_idx % PROBE_INTERVAL_FRAMES == 0 or frame_idx in PROBE_FRAMES_SET):
            print(f"[probe] pre_idx={frame_idx} pre_ms={pre_ms:.3f}  post_idx={post_idx} post_ms={post_ms:.3f}")

        if frame_idx >= end_frame:
            break

        if ret and frame_idx % FRAME_INTERVAL == 0:
            # 提前檢查暗房區間，避免不必要的計算
            current_time_seconds = frame_idx / fps
            is_darkroom, darkroom_info = is_in_darkroom_interval(current_time_seconds, darkroom_intervals_seconds)

            if is_darkroom:
                # 暗房區間：跳過所有運動計算，直接記錄零值結果
                # 仍需維護幀緩存以便物理群集檢測
                frame_cache.append((frame_idx, frame.copy()))
                if len(frame_cache) > 20:
                    frame_cache.pop(0)

                result['frame'].append(frame)
                result['frame_idx'].append(frame_idx)
                result['keypoints'].append([])
                result['kp_pair_lines'].append([])
                result['camera_pan'].append(True)  # 標記為類似camera_pan
                result['v_travel_distance'].append(0)
                result['frame_path'].append('')  # 暗房區間無匯出圖片
                continue

            # 檢查是否需要旋轉影像
            if file_name in rotation_config:
                rotation_angle = rotation_config[file_name]
                frame = rotate_frame(frame, rotation_angle)

            keypoint_list2, feature_descrpitor2 = feature_detector.detectAndCompute(frame, mask)

            # default values
            vertical_travel_distance = 0
            camera_pan = False
            display_keypoints = []
            kp_pair_lines = []

            if feature_descrpitor1 is not None and feature_descrpitor2 is not None:
                matches = feature_matcher.match(feature_descrpitor2, feature_descrpitor1)
                paired_keypoints_info_array = []

                for match_info in matches:
                    kp1_idx = match_info.trainIdx
                    kp2_idx = match_info.queryIdx
                    kp1_coord = np.array(keypoint_list1[kp1_idx].pt, dtype=int)
                    kp2_coord = np.array(keypoint_list2[kp2_idx].pt, dtype=int)
                    paired_keypoints_info_array.append([
                        kp2_idx, 
                        kp1_idx, 
                        np.sqrt(np.sum((kp1_coord - kp2_coord)**2)),
                        kp2_coord[0]-kp1_coord[0], # horizontal travel distance
                        kp2_coord[1]-kp1_coord[1], # vertical travel distance
                    ])

                paired_keypoints_info_array = np.array(paired_keypoints_info_array)
                paired_keypoints_info_array = paired_keypoints_info_array[utils.remove_outlier_idx(paired_keypoints_info_array[:, 2], 'upper')]

                if paired_keypoints_info_array.shape[0] > 1:
                    
                    display_keypoints = [keypoint_list2[kp2_idx] for kp2_idx in paired_keypoints_info_array[:, 0].astype(int)]
                    kp_pair_lines = [(np.array(keypoint_list1[int(kp1_idx)].pt, dtype=int), np.array(keypoint_list2[int(kp2_idx)].pt, dtype=int)) \
                                     for kp2_idx, kp1_idx in paired_keypoints_info_array[:, :2]]
                    camera_pan = ttest_1samp(paired_keypoints_info_array[:, 3], 0).pvalue < 0.001
                    group_idx_array = cluster.fit_predict(paired_keypoints_info_array[:, 2].reshape(-1, 1))

                    if len(set(group_idx_array)) > 1 and camera_pan == False:
                        group0_v_travel_array = paired_keypoints_info_array[np.where(group_idx_array==0)[0], 4]
                        group1_v_travel_array = paired_keypoints_info_array[np.where(group_idx_array==1)[0], 4]

                        group0_v_travel = np.median(group0_v_travel_array) if ttest_1samp(group0_v_travel_array, 0).pvalue < 0.0001 else 0
                        group1_v_travel = np.median(group1_v_travel_array) if ttest_1samp(group1_v_travel_array, 0).pvalue < 0.0001 else 0

                        if abs(group0_v_travel) > abs(group1_v_travel):
                            vertical_travel_distance = int(group1_v_travel - group0_v_travel)
                        else:
                            vertical_travel_distance = int(group0_v_travel - group1_v_travel)
                    else:
                        vertical_travel_distance = 0
            
            # 暗房區間檢查已移至前面，此處不再需要

            # 維護幀緩存（保留最近20幀）
            frame_cache.append((frame_idx, frame.copy()))
            if len(frame_cache) > 20:
                frame_cache.pop(0)

            # 檢查是否有有效的比例尺資料（提前計算以便物理群集檢測）
            if file_name in video_scale_dict:
                scale_factor = video_scale_dict[file_name]
            else:
                print(f"⚠️  警告: 影片 {file_name} 沒有有效的比例尺資料，使用預設值 1.0")
                scale_factor = 1.0

            # 物理群集檢測與PNG匯出邏輯
            frame_path = ''  # 默認空標籤

            # 轉換為毫米的運動距離
            v_travel_distance_mm = vertical_travel_distance * 10 / scale_factor

            if v_travel_distance_mm != 0 and not in_physical_cluster:
                # 開始新的物理群集
                physical_cluster_counter += 1
                current_cluster_id = physical_cluster_counter
                in_physical_cluster = True

                # 標記前一幀為前0點（如果有的話）
                if len(result['frame_path']) > 0:
                    result['frame_path'][-1] = f'pre_cluster_{current_cluster_id:03d}.jpg'
                    # 記錄待匯出的前0點
                    if len(frame_cache) >= 2:
                        pending_pre_export = (frame_cache[-2], f'pre_cluster_{current_cluster_id:03d}.jpg')
                else:
                    # 第一幀就有運動：使用第一幀作為前0點（特殊情況）
                    print(f"⚠️ 物理群集 {current_cluster_id} 從第一幀開始，使用第一幀作為前0點")
                    if len(frame_cache) >= 1:
                        pending_pre_export = (frame_cache[-1], f'pre_cluster_{current_cluster_id:03d}.jpg')

            elif v_travel_distance_mm == 0 and in_physical_cluster:
                # 物理群集結束，標記當前幀為後0點
                frame_path = f'post_cluster_{current_cluster_id:03d}.jpg'
                in_physical_cluster = False

                # 匯出前0點和後0點JPG
                video_name = os.path.splitext(file_name)[0]
                if pending_pre_export:
                    export_frame_jpg(pending_pre_export[0], pending_pre_export[1], video_name)
                    pending_pre_export = None

                export_frame_jpg((frame_idx, frame), frame_path, video_name)
                current_cluster_id = None

            result['frame'].append(frame)
            result['frame_idx'].append(frame_idx)
            result['keypoints'].append(display_keypoints)
            result['kp_pair_lines'].append(kp_pair_lines)
            result['camera_pan'].append(camera_pan)  # camera_pan 判斷
            result['v_travel_distance'].append(v_travel_distance_mm)
            result['frame_path'].append(frame_path)  # 由物理群集檢測設定的標籤

            keypoint_list1 = keypoint_list2
            feature_descrpitor1 = feature_descrpitor2

    # 處理掃描結束時仍在進行中的物理群集
    if in_physical_cluster:
        video_name = os.path.splitext(file_name)[0]

        # 匯出前0點（如果有的話）
        if pending_pre_export:
            export_frame_jpg(pending_pre_export[0], pending_pre_export[1], video_name)

        # 使用最後一幀作為後0點（特殊情況）
        if len(frame_cache) > 0:
            last_frame_data = frame_cache[-1]
            post_jpg_filename = f'post_cluster_{current_cluster_id:03d}.jpg'
            export_frame_jpg(last_frame_data, post_jpg_filename, video_name)

            # 更新最後一個 frame_path
            if len(result['frame_path']) > 0:
                result['frame_path'][-1] = post_jpg_filename

            print(f"⚠️ 物理群集 {current_cluster_id} 在掃描結束時仍未完成，使用最後一幀作為後0點")
        else:
            print(f"❌ 物理群集 {current_cluster_id} 在掃描結束時無法找到後0點參考幀")

    # post-process the result
    for idx in range(1, len(result['v_travel_distance'])-1):
        if result['v_travel_distance'][idx] != 0 and (result['v_travel_distance'][idx-1]==0 and result['v_travel_distance'][idx+1]==0):
            result['v_travel_distance'][idx] = 0

    # original video reset to frame 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path.replace("data", "inspection"), fourcc, fps/FRAME_INTERVAL, (w, h))

    travel_distance_sum = 0

    for frame, frame_idx, keypoints, kp_pair_lines, camera_pan, vertical_travel_distance in zip(
        result['frame'], result['frame_idx'], result['keypoints'], result['kp_pair_lines'], result['camera_pan'], result['v_travel_distance']):
        travel_distance_sum += vertical_travel_distance

        # 檢查當前幀是否在暗房區間內（用於顯示）
        current_time_seconds = frame_idx / fps
        is_darkroom, _ = is_in_darkroom_interval(current_time_seconds, darkroom_intervals_seconds)

        # draw the display info
        frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        for coord1, coord2 in kp_pair_lines:
            cv2.line(frame, coord1, coord2, [0, 0, 255], 2)
        
        # 決定顯示文字和顏色
        if is_darkroom:
            display_text = "darkroom (ignored)"
            text_color = (128, 128, 128)  # 灰色
        elif camera_pan and not is_darkroom:
            display_text = "camera pan"
            text_color = (0, 255, 255)  # 黃色
        else:
            display_text = f"travel: {round(travel_distance_sum, 5)} mm"
            text_color = (0, 0, 255) if vertical_travel_distance == 0 else (0, 255, 0)  # 紅色/綠色
        
        # 顯示 Frame ID (第一行)
        cv2.putText(
            frame, 
            f"Frame: {frame_idx}", 
            (10, h-110), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255),  # 白色
            2)
        
        # 顯示時間和狀態信息 (第二行)
        cv2.putText(
            frame, 
            f"{round(frame_idx/fps, 1)} sec  {display_text}", 
            (10, h-80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            text_color, 
            2)
        
        out.write(frame)
            
    out.release()

    # write down the record
    # 正確生成 CSV 檔案路徑
    video_filename = os.path.basename(video_path)  # 取得檔名 (例如: 21.mp4)
    csv_filename = os.path.splitext(video_filename)[0] + ".csv"  # 移除副檔名並加上 .csv (例如: 21.csv)
    csv_path = os.path.join(DATA_FOLDER, 'lifts', 'result', csv_filename)
    
    print(f"💾 儲存 CSV 檔案: {csv_path}")
    
    pd.DataFrame({
        'frame_idx':result['frame_idx'],
        'second':[round(i/(fps), 3) for i in result['frame_idx']],
        'vertical_travel_distance (mm)':result['v_travel_distance'],
        'frame_path':result['frame_path']
    }).to_csv(csv_path, index=False)

    print(f"complete: {video_path}")


# 比例尺處理 - 使用快取機制
scale_images_dir = os.path.join(DATA_FOLDER, 'lifts', 'scale_images')

print("📏 載入比例尺快取...")
scale_cache, cache_info = load_scale_cache()

# 檢查快取是否有效
cache_valid = is_cache_valid(scale_images_dir, cache_info)

# 取得所有影片檔案
video_files = []
for root, folder, files in os.walk(os.path.join(DATA_FOLDER, 'lifts', 'data')):
    video_files.extend([f for f in files if f.endswith('.mp4')])

# 確定需要計算比例尺的影片
if cache_valid:
    missing_videos = get_missing_videos(scale_cache, video_files)
    print(f"📋 快取有效，發現 {len(missing_videos)} 個新影片需要計算比例尺")
else:
    missing_videos = video_files
    scale_cache = {}
    print("🔄 快取無效或不存在，需要重新計算所有比例尺")

print_cache_status(scale_cache, missing_videos)

# 只處理需要計算的影片對應的比例尺圖片
new_scale_data = {}
for root, folder, files in os.walk(scale_images_dir):
    for file in files:
        video_name = "-".join(file.split(sep="-")[:-1]) + ".mp4"
        
        # 只處理缺少快取的影片
        if video_name not in missing_videos:
            continue
        
        print(f"🔄 處理比例尺圖片: {file} (影片: {video_name})")
        image = cv2.imread(os.path.join(root, file))
        
        # 從原始圖片中尋找紅色標記點
        filtered_array = (image[..., 0] < 10) * (image[..., 1] < 10) * (image[..., 2] > 250)
        points = np.where(filtered_array)
        
        # 檢查是否找到足夠的紅色標記點
        if len(points[0]) < 2:
            image_path = os.path.join(root, file)
            print(f"❌ 比例尺錯誤: 在圖片 '{image_path}' 中找不到足夠的紅色標記點")
            print(f"   對應影片: {video_name}")
            print(f"   找到紅點數量: {len(points[0])} (需要至少 2 個)")
            print(f"   請檢查並重新標記紅色點")
            continue
        
        # 取得兩個紅點的座標 (y, x)
        point1_original = (points[0][0], points[1][0])  # (y1, x1)
        point2_original = (points[0][1], points[1][1])  # (y2, x2)
        
        # 計算原始歐氏距離（作為驗算基準）
        original_euclidean_distance = np.sqrt(
            (point1_original[0] - point2_original[0])**2 + 
            (point1_original[1] - point2_original[1])**2
        )
        
        # 複製座標用於旋轉計算
        point1 = point1_original
        point2 = point2_original
        
        # 如果影片需要旋轉，對紅點座標進行相應的旋轉變換
        if video_name in rotation_config:
            rotation_angle = rotation_config[video_name]
            print(f"  🔄 旋轉比例尺座標 (角度: {rotation_angle}°)")
            
            # 取得圖片中心點
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # 將角度轉換為弧度
            angle_rad = np.radians(rotation_angle)
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)
            
            # 對兩個點進行旋轉變換
            def rotate_point(y, x, center_y, center_x, cos_a, sin_a):
                # 將座標移至原點
                rel_x = x - center_x
                rel_y = y - center_y
                # 進行旋轉（注意座標系統：影像 y 軸向下）
                new_x = rel_x * cos_a + rel_y * sin_a
                new_y = -rel_x * sin_a + rel_y * cos_a
                # 移回原位置
                return new_y + center_y, new_x + center_x
            
            point1 = rotate_point(point1[0], point1[1], center_y, center_x, cos_angle, sin_angle)
            point2 = rotate_point(point2[0], point2[1], center_y, center_x, cos_angle, sin_angle)
            
            # 旋轉驗算
            distance = abs(point1[0] - point2[0])
            difference_ratio = abs(distance - original_euclidean_distance) / original_euclidean_distance
            difference_percent = difference_ratio * 100
            
            if difference_percent > 10.0:
                print(f"    ⚠️  旋轉驗算警告: 差異 {difference_percent:.1f}% (檔案: {file})")
        else:
            # 沒有旋轉時，使用原始垂直距離
            distance = abs(point1_original[0] - point2_original[0])
        
        # 最終使用的垂直方向距離
        final_distance = abs(point1[0] - point2[0])
        
        print(f"  ✅ 垂直距離: {final_distance:.2f} 像素")
        
        if video_name in new_scale_data:
            new_scale_data[video_name].append(final_distance)
        else:
            new_scale_data[video_name] = [final_distance]

# 計算平均值並更新快取
for video_name, distances in new_scale_data.items():
    scale_cache[video_name] = np.mean(distances)
    print(f"📊 {video_name}: 平均比例尺 {scale_cache[video_name]:.4f} 像素")

# 儲存更新的快取
if new_scale_data:
    cache_info = {
        'last_updated': datetime.datetime.now().isoformat(),
        'total_videos': len(scale_cache),
        'directory_hash': generate_scale_images_hash(scale_images_dir),
        'newly_processed': len(new_scale_data)
    }
    save_scale_cache(scale_cache, cache_info)

# 設定最終的比例尺字典供主程式使用
video_scale_dict = scale_cache

print(f"\n📊 比例尺處理完成，共處理 {len(video_scale_dict)} 個影片的比例尺資料:")
for video, scale in video_scale_dict.items():
    print(f"  {video}: {scale:.2f} 像素")

print(f"\n🎬 開始處理影片...")

for root, folder, files in os.walk(os.path.join(DATA_FOLDER, 'lifts','data')):
    for file in files:
        if TARGET_VIDEO and file != TARGET_VIDEO:
            continue
        print(f"\n🎥 正在處理影片: {file}")
        scan(os.path.join(root, file), file)
        print(f"✅ 影片處理完成: {file}")

print(f"\n🎉 所有影片處理完成！")


# scan(os.path.join(DATA_FOLDER, "lifts", "data", "2.mp4"), "2.mp4")

