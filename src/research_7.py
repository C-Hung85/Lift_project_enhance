'''reference
https://blog.csdn.net/weixin_43151193/article/details/125222481
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
'''

import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import warnings
import cv2
import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool
from config import Config

warnings.filterwarnings('ignore') 

orb = cv2.ORB_create(nfeatures=100)
bf_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
cluster = KMeans(n_clusters=2)
interval = Config['scan_setting']['interval']
display = True

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts','data')):
    for file in files:
        path_list.append(os.path.join(root, file))

path = "/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4"

def job(path):
    vidcap = cv2.VideoCapture(path)
    ret, frame = vidcap.read()
    kp1, des1 = orb.detectAndCompute(frame, None)
    frame = cv2.drawKeypoints(frame, kp1, None, color=(0, 255, 0), flags=0)
    frame_idx = 0
    h, w = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, vidcap.get(cv2.CAP_PROP_FPS)/interval, (w, h))
    out.write(frame)

    while ret:
        ret, frame = vidcap.read()
        frame_idx += 1

        if ret and frame_idx % interval == 0:
            kp2, des2 = orb.detectAndCompute(frame, None)
            
            matches = bf_matcher.match(des2, des1)
            kp_idx_dist_array = []

            for match_info in matches:
                kp1_idx = match_info.trainIdx
                kp2_idx = match_info.queryIdx
                kp1_coord = np.array(kp1[kp1_idx].pt, dtype=int)
                kp2_coord = np.array(kp2[kp2_idx].pt, dtype=int)
                kp_idx_dist_array.append([kp2_idx, kp1_idx, np.sqrt(np.sum((kp1_coord - kp2_coord)**2))])

            kp_idx_dist_array = np.array(kp_idx_dist_array)
            distance_q1, distance_q3 = np.quantile(kp_idx_dist_array[:,2], 0.25), np.quantile(kp_idx_dist_array[:,2], 0.75)
            valid_kp_idx_dist_array = kp_idx_dist_array[np.where(kp_idx_dist_array[:,2] <= distance_q3+1.5*(distance_q3-distance_q1))[0]]
            valid_kp_list = [kp2[i] for i in valid_kp_idx_dist_array[:, 0].astype(int)]

            for kp2_idx, kp1_idx, distance in valid_kp_idx_dist_array:
                cv2.line(frame, np.array(kp1[int(kp1_idx)].pt, dtype=int), np.array(kp2[int(kp2_idx)].pt, dtype=int), [0, 0, 255], 2)
            frame = cv2.drawKeypoints(frame, valid_kp_list, None, color=(0, 255, 0), flags=0)

            group_idx_array = cluster.fit_predict(valid_kp_idx_dist_array[:,2].reshape(-1,1))
            if len(set(group_idx_array)) > 1:
                group0_move_distance = np.mean(valid_kp_idx_dist_array[np.where(group_idx_array==0)[0], 2])
                group1_move_distance = np.mean(valid_kp_idx_dist_array[np.where(group_idx_array==1)[0], 2])
                distance_diff = abs(group0_move_distance - group1_move_distance)
            else:
                distance_diff = 0

            cv2.putText(frame, str(distance_diff), (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
            kp1 = kp2
            des1 = des2
            
    out.release()
    print(f"complete: {path}")

# with Pool(5) as pool:
#     pool.map(job, path_list)

job(path)