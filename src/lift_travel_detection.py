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

warnings.filterwarnings('ignore')

# Parameters and objects
DATA_FOLDER = Config['files']['data_folder']
feature_detector = cv2.ORB.create(nfeatures=100)
feature_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING, crossCheck=True)
cluster = KMeans(n_clusters=2)
FRAME_INTERVAL = Config['scan_setting']['interval']
ROI_RATIO = 0.6

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

    # create a mask to define the ROI
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h*ROI_RATIO/2):int(h*(1-ROI_RATIO/2)), int(w*ROI_RATIO/2):int(w*(1-ROI_RATIO/2))] = 1

    # record container
    result = {
        'frame':[],
        'frame_idx':[],
        'keypoints':[], 
        'camera_pan':[],
        'v_travel_distance':[],
        'kp_pair_lines':[]
    }

    # detect keypoints
    keypoint_list1, feature_descrpitor1 = feature_detector.detectAndCompute(frame, mask)

    # set video to the start point
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while ret:
        frame_idx = int(vidcap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = vidcap.read()

        if frame_idx >= end_frame:
            break

        if ret and frame_idx % FRAME_INTERVAL == 0:
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
            
            result['frame'].append(frame)
            result['frame_idx'].append(frame_idx)
            result['keypoints'].append(display_keypoints)
            result['kp_pair_lines'].append(kp_pair_lines)
            result['camera_pan'].append(camera_pan)
            result['v_travel_distance'].append(vertical_travel_distance * 10 / video_scale_dict[file_name])

            keypoint_list1 = keypoint_list2
            feature_descrpitor1 = feature_descrpitor2
    
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

        # read the indicated frame from the original video
        # vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # ret, frame = vidcap.read()

        # draw the display info
        frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)
        for coord1, coord2 in kp_pair_lines:
            cv2.line(frame, coord1, coord2, [0, 0, 255], 2)
        
        cv2.putText(
            frame, 
            f"""{round(frame_idx/fps, 1)} sec  {"camera pan" if camera_pan else f"travel: {round(travel_distance_sum, 5)} mm"}""", 
            (10, h-80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 255) if camera_pan else ((0, 0, 255) if vertical_travel_distance==0 else (0, 255, 0)), 
            2)
        
        out.write(frame)
            
    out.release()

    # write down the record
    pd.DataFrame({
        'second':[round(i/(fps), 3) for i in result['frame_idx']],
        'vertical_travel_distance (mm)':result['v_travel_distance']
    }).to_csv(video_path.replace("data", "result").split(sep=".")[0]+".csv", index=False)

    print(f"complete: {video_path}")


video_scale_dict = {}
for root, folder, files in os.walk(os.path.join(DATA_FOLDER, 'lifts', 'scale_images')):
    for file in files:
        video_name = "-".join(file.split(sep="-")[:-1]) + ".mp4"
        image = cv2.imread(os.path.join(os.path.join(root, file)))
        filtered_array = (image[..., 0] < 10) * (image[..., 1] < 10) * (image[..., 2] > 250)
        points = np.where(filtered_array)
        distance = np.sqrt((points[0][0] - points[0][1])**2 + (points[1][0] - points[1][1])**2)
        if video_name in video_scale_dict:
            video_scale_dict[video_name].append(distance)
        else:
            video_scale_dict[video_name] = [distance]

video_scale_dict = {video:np.mean(values) for video, values in video_scale_dict.items()}


for root, folder, files in os.walk(os.path.join(DATA_FOLDER, 'lifts','data')):
    for file in files:
        scan(os.path.join(root, file), file)


# scan(os.path.join(DATA_FOLDER, "lifts", "data", "2.mp4"), "2.mp4")

