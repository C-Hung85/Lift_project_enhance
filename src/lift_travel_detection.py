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
    """匯出單個幀為JPG（於 exported_frames/<video_name>/ 下）"""
    frame_idx, frame = frame_data

    export_dir = os.path.join('lifts', 'exported_frames', video_name)
    os.makedirs(export_dir, exist_ok=True)

    export_path = os.path.join(export_dir, jpg_filename)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
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
cluster = KMeans(n_clusters=2, random_state=0)
FRAME_INTERVAL = Config['scan_setting']['interval']
ROI_RATIO = 0.6
# 檢測參數（像素域）
EFFECT_MIN_PX = 1.0
PAIR_TOLERANCE_PX = 1.0
JITTER_MAX_PX = 1.0
EXIT_ZERO_LEN = 3
REVERSAL_PERSIST_R = 2
MIN_MATCHES = 6


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
        'frame_path':[],
        # 腳手架：之後將填入真正的群集資訊；此階段先固定為 0/空字串
        'cluster_id':[],      # 群外為 0
        'orientation':[],     # -1/0/+1；群外 0
        'darkroom_event':[]   # 'enter_darkroom' / 'exit_darkroom' / ''
    }

    # 狀態機變數（群集與抖動處理，入群延遲一幀）
    state = 'Idle'  # Idle / PendingEnter / InCluster
    pending_idx = None
    pending_delta_px = None
    pending_result_idx = None
    current_cluster_id = 0
    physical_cluster_counter = 0
    orientation_current = 0
    zero_streak = 0
    reversal_streak = 0
    # 匯出/快取變數
    frame_cache = []                    # 最近處理幀 (frame_idx, frame)
    pending_pre_export = None           # (frame_data, jpg_filename)
    last_non_darkroom_frame = None      # (frame_idx, frame)
    video_name = os.path.splitext(file_name)[0]

    # detect keypoints
    keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame, mask)

    # set video to the start point
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_is_darkroom = False

    while ret:
        frame_idx = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = vidcap.read()

        if frame_idx >= end_frame:
            break

        if ret and frame_idx % FRAME_INTERVAL == 0:
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

                if paired_keypoints_info_array.shape[0] >= MIN_MATCHES:
                    
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
            
            # 檢查是否在暗房區間內，如果是則忽略運動（類似 camera pan）
            current_time_seconds = frame_idx / fps
            is_darkroom, darkroom_info = is_in_darkroom_interval(current_time_seconds, darkroom_intervals_seconds)
            
            # 如果在暗房區間內，將運動距離設為 0（忽略）
            if is_darkroom:
                vertical_travel_distance = 0

            # 暗房事件（僅記錄進出，不改變顯示文字行為）
            if is_darkroom and not prev_is_darkroom:
                darkroom_event = 'enter_darkroom'
            elif (not is_darkroom) and prev_is_darkroom:
                darkroom_event = 'exit_darkroom'
            else:
                darkroom_event = ''
            
            # 狀態機計算（以像素域判斷候選）
            delta_px = vertical_travel_distance
            is_candidate = (not camera_pan) and (abs(delta_px) >= EFFECT_MIN_PX) and (not is_darkroom)
            effective_mm_current = 0.0

            if is_darkroom:
                # 暗房：強制回到 Idle 狀態，不累計群集
                if state == 'InCluster':
                    # 在群內時進入暗房 → 強制出群，post 用暗房前最後一個非暗房幀
                    if last_non_darkroom_frame is not None and current_cluster_id:
                        post_name = f'post_cluster_{current_cluster_id:03d}.jpg'
                        export_frame_jpg(last_non_darkroom_frame, post_name, video_name)
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

            elif state == 'Idle':
                zero_streak = 0
                reversal_streak = 0
                if is_candidate:
                    state = 'PendingEnter'
                    pending_idx = frame_idx
                    pending_delta_px = delta_px
                    pending_result_idx = len(result['frame_idx'])  # 將在本迭代末尾寫入 0，之後再回填
                # Idle 狀態輸出 0

            elif state == 'PendingEnter':
                if not is_candidate:
                    # 下一幀不是候選 → 視為噪聲，取消 pending
                    state = 'Idle'
                    pending_idx = None
                    pending_delta_px = None
                    pending_result_idx = None
                else:
                    # 是候選：檢查符號關係
                    if np.sign(delta_px) != np.sign(pending_delta_px):
                        # 相反號：先檢查是否為可抵銷的正負對
                        if abs(delta_px + pending_delta_px) <= PAIR_TOLERANCE_PX:
                            # 典型正負對 → 抵銷後回 Idle
                            state = 'Idle'
                            pending_idx = None
                            pending_delta_px = None
                            pending_result_idx = None
                        else:
                            # 邊界相反但不成對 → 將當前候選改為新的 pending，等待下一幀再決定
                            pending_idx = frame_idx
                            pending_delta_px = delta_px
                            pending_result_idx = len(result['frame_idx'])
                            # 保持 PendingEnter 狀態
                    else:
                        # 同號候選 → 確認入群（以前一幀 pending 為起點）
                        state = 'InCluster'
                        physical_cluster_counter += 1
                        current_cluster_id = physical_cluster_counter
                        orientation_current = 1 if pending_delta_px > 0 else -1
                        # 標記 pre：將 pending 那幀的 frame_path 設為 pre，並準備匯出快照
                        pre_name = f'pre_cluster_{current_cluster_id:03d}.jpg'
                        if pending_result_idx is not None and pending_result_idx < len(result['frame_path']):
                            result['frame_path'][pending_result_idx] = pre_name
                        # 從 frame_cache 取對應影格（優先 -2，其次 -1）
                        pre_frame = frame_cache[-2] if len(frame_cache) >= 2 else (frame_idx, frame)
                        pending_pre_export = (pre_frame, pre_name)
                        # 回填 pending 幀的 mm、cluster 與方向
                        if pending_result_idx is not None and pending_result_idx < len(result['v_travel_distance']):
                            scale_factor = video_scale_dict.get(file_name, 1.0)
                            result['v_travel_distance'][pending_result_idx] = pending_delta_px * 10 / scale_factor
                            result['cluster_id'][pending_result_idx] = current_cluster_id
                            result['orientation'][pending_result_idx] = orientation_current
                        pending_idx = None
                        pending_delta_px = None
                        pending_result_idx = None

            elif state == 'InCluster':
                if is_candidate:
                    if np.sign(delta_px) != orientation_current and abs(delta_px) <= JITTER_MAX_PX:
                        # 小幅反向抖動：視為 0（不改狀態）
                        pass
                    elif np.sign(delta_px) != orientation_current:
                        reversal_streak += 1
                        if reversal_streak >= REVERSAL_PERSIST_R:
                            # 真反轉：關閉當前群，下一幀再重新 Pending（簡化：回 Idle）
                            state = 'Idle'
                            current_cluster_id = 0
                            orientation_current = 0
                            reversal_streak = 0
                            zero_streak = 0
                        else:
                            # 暫時視為 0
                            pass
                    else:
                        # 同向：維持
                        reversal_streak = 0
                        zero_streak = 0
                        effective_mm_current = delta_px * 10 / video_scale_dict.get(file_name, 1.0)
                else:
                    zero_streak += 1
                    if zero_streak >= EXIT_ZERO_LEN:
                        # 出群
                        # 匯出 pre（若尚未匯出）與 post
                        if current_cluster_id:
                            post_name = f'post_cluster_{current_cluster_id:03d}.jpg'
                            export_frame_jpg((frame_idx, frame), post_name, video_name)
                            if pending_pre_export is not None:
                                export_frame_jpg(pending_pre_export[0], pending_pre_export[1], video_name)
                                pending_pre_export = None
                            # 記錄本幀 post 標記（設當前索引）
                            current_index = len(result['frame_idx'])
                            # 本幀稍後會被 append；因此先記個變數，稍後補寫
                            post_mark_name = post_name
                        else:
                            post_mark_name = ''
                        state = 'Idle'
                        current_cluster_id = 0
                        orientation_current = 0
                        zero_streak = 0
                        reversal_streak = 0
                    else:
                        # 仍在群內但本幀不是候選 → 輸出 0
                        pass

            # 維護幀快取
            frame_cache.append((frame_idx, frame.copy()))
            if len(frame_cache) > 20:
                frame_cache.pop(0)
            if not is_darkroom:
                last_non_darkroom_frame = (frame_idx, frame.copy())

            # 寫入結果（目前不計 mm 位移為 0 的濾除，維持原行為）
            result['frame'].append(frame)
            result['frame_idx'].append(frame_idx)
            result['keypoints'].append(display_keypoints)
            result['kp_pair_lines'].append(kp_pair_lines)
            result['camera_pan'].append(camera_pan or is_darkroom)  # camera_pan 或暗房區間都顯示為 pan
            # 檢查是否有有效的比例尺資料
            if file_name in video_scale_dict:
                scale_factor = video_scale_dict[file_name]
            else:
                print(f"⚠️  警告: 影片 {file_name} 沒有有效的比例尺資料，使用預設值 1.0")
                scale_factor = 1.0
            # 依狀態輸出當前幀的有效位移
            if state == 'InCluster' and effective_mm_current == 0.0 and is_candidate and np.sign(delta_px) == orientation_current:
                effective_mm_current = delta_px * 10 / scale_factor
            result['v_travel_distance'].append(effective_mm_current)
            result['cluster_id'].append(current_cluster_id if state == 'InCluster' else 0)
            result['orientation'].append(orientation_current if state == 'InCluster' else 0)
            # 腳手架：先寫入空的群集資訊（之後實作狀態機再填實）
            result['darkroom_event'].append(darkroom_event)
            # 預設填空 frame_path
            result['frame_path'].append('')
            # 若剛剛決定 post（出群），把本幀標記為 post
            if 'post_mark_name' in locals() and post_mark_name:
                result['frame_path'][-1] = post_mark_name
                del post_mark_name

            keypoint_list1 = keypoint_list2
            feature_descrpitor1 = feature_descrpitor2
            prev_is_darkroom = is_darkroom
    
    # post-process the result
    for idx in range(1, len(result['v_travel_distance'])-1):
        if result['v_travel_distance'][idx] != 0 and (result['v_travel_distance'][idx-1]==0 and result['v_travel_distance'][idx+1]==0):
            result['v_travel_distance'][idx] = 0

    # original video reset to frame 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path.replace("data", "inspection"), fourcc, fps/FRAME_INTERVAL, (w, h))

    travel_distance_sum = 0

    for frame, frame_idx, keypoints, kp_pair_lines, camera_pan, vertical_travel_distance, cluster_id_disp, orientation_disp in zip(
        result['frame'], result['frame_idx'], result['keypoints'], result['kp_pair_lines'], result['camera_pan'], result['v_travel_distance'], result['cluster_id'], result['orientation']):
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
        
        # 顯示 Frame ID（第一行）
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, h-110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2)

        # 顯示時間與狀態（第二行）
        cv2.putText(
            frame, 
            f"{round(frame_idx/fps, 1)} sec  {display_text}", 
            (10, h-80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            text_color, 
            2)

        # 顯示群集與方向（第三行，僅在群內）
        if cluster_id_disp:
            arrow = '↑' if orientation_disp > 0 else ('↓' if orientation_disp < 0 else '')
            cv2.putText(
                frame,
                f"cluster #{cluster_id_disp} {arrow}",
                (10, h-50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
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
        'cluster_id':result['cluster_id'],
        'orientation':result['orientation'],
        'darkroom_event':result['darkroom_event'],
        'frame_path':result.get('frame_path', ['']*len(result['frame_idx']))
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
        print(f"\n🎥 正在處理影片: {file}")
        scan(os.path.join(root, file), file)
        print(f"✅ 影片處理完成: {file}")

print(f"\n🎉 所有影片處理完成！")


# scan(os.path.join(DATA_FOLDER, "lifts", "data", "2.mp4"), "2.mp4")

