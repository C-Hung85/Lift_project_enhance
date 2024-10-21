import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import warnings
import utils
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans
from multiprocessing import Pool
from config import Config

warnings.filterwarnings('ignore')

# Parameters and objects
orb = cv2.ORB.create(nfeatures=60)
bf_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING, crossCheck=True)
cluster = KMeans(n_clusters=2)
FRAME_INTERVAL = Config['scan_setting']['interval']
ROI_RATIO = 0.5

# create necessary folders
for folder_name in ['inspection', 'result']:
    os.makedirs(folder_name, exist_ok=True)

def scan(video_path):
    vidcap = cv2.VideoCapture(video_path)
    ret, frame = vidcap.read()
    h, w = frame.shape[:2]
    fps = vidcap.get(cv2.CAP_PROP_FPS)/FRAME_INTERVAL
    frame_idx = 0

    # create a mask to define the ROI
    mask = np.zeros((h, w), dtype=np.int8)
    mask[int(h*ROI_RATIO/2):int(h*(1-ROI_RATIO/2)), int(w*ROI_RATIO/2):int(w*(1-ROI_RATIO/2))] = 1

    # detect keypoints
    keypoint_list1, feature_descrpitor1 = orb.detectAndCompute(frame, mask)
    frame = cv2.drawKeypoints(frame, keypoint_list1, None, color=(0, 255, 0), flags=0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path.replace("data", "result"), fourcc, fps, (w, h))
    out.write(frame)

    while ret:
        ret, frame = vidcap.read()
        frame_idx += 1

        if ret and frame_idx % FRAME_INTERVAL == 0:
            keypoint_list2, feature_descrpitor2 = orb.detectAndCompute(frame, mask)
            vertical_travel_distance = 0
            camera_pan = False

            if feature_descrpitor1 is not None and feature_descrpitor2 is not None:
                matches = bf_matcher.match(feature_descrpitor2, feature_descrpitor1)
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

                    frame = cv2.drawKeypoints(
                        frame, 
                        [keypoint_list2[i] for i in paired_keypoints_info_array[:, 0].astype(int)], 
                        None, 
                        color=(0, 255, 0), 
                        flags=0)
                    
                    camera_pan = ttest_1samp(paired_keypoints_info_array[:, 3], 0).pvalue < 0.05

                    if camera_pan == False:
                        group_idx_array = cluster.fit_predict(paired_keypoints_info_array[:, 2].reshape(-1, 1))

                        if len(set(group_idx_array)) > 1:
                            group0_v_travel_array = paired_keypoints_info_array[np.where(group_idx_array==0)[0], 4]
                            group1_v_travel_array = paired_keypoints_info_array[np.where(group_idx_array==1)[0], 4]

                            group0_v_travel = 0 if ttest_1samp(group0_v_travel_array, 0).pvalue > 0.1 else np.median(group0_v_travel_array)
                            group1_v_travel = 0 if ttest_1samp(group1_v_travel_array, 0).pvalue > 0.1 else np.median(group1_v_travel_array)

                            if abs(group0_v_travel) > abs(group1_v_travel):
                                vertical_travel_distance = group1_v_travel - group0_v_travel
                            else:
                                vertical_travel_distance = group0_v_travel - group1_v_travel
            
            cv2.putText(
                frame, 
                "camera pan" if camera_pan else f"pixel travel: {vertical_travel_distance} pixels", 
                (10, h-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 255) if camera_pan else ((0, 0, 255) if group1_v_travel==0 else (0, 255, 0)), 
                2)
            
            out.write(frame)
            keypoint_list1 = keypoint_list2
            feature_descrpitor1 = feature_descrpitor2
            
    out.release()
    print(f"complete: {video_path}")


path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts','data')):
    for file in files:
        path_list.append(os.path.join(root, file))

with Pool(2) as pool:
    pool.map(scan, path_list)
