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
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans
from multiprocessing import Pool
from config import Config

warnings.filterwarnings('ignore') 

feature_detector = cv2.ORB.create(nfeatures=60)
feature_matcher = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING, crossCheck=True)
# feature_matcher = cv2.FlannBasedMatcher({'algorithm':6, 'table_number':6, 'key_size':12, 'multi_probe_level':1}, {})
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
    kp1, des1 = feature_detector.detectAndCompute(frame, None)
    frame = cv2.drawKeypoints(frame, kp1, None, color=(0, 255, 0), flags=0)
    frame_idx = 0
    h, w = frame.shape[:2]
    fps = vidcap.get(cv2.CAP_PROP_FPS)/interval

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, fps, (w, h))
    out.write(frame)

    while ret:
        ret, frame = vidcap.read()
        frame_idx += 1

        if ret and frame_idx % interval == 0:
            kp2, des2 = feature_detector.detectAndCompute(frame, None)
            frame = cv2.drawKeypoints(frame, kp2, None, color=(0, 255, 0), flags=0)

            if des2 is not None and des1 is not None:
                matches = feature_matcher.match(des2, des1)

                for match_info in matches:
                    kp1_idx = match_info.trainIdx
                    kp2_idx = match_info.queryIdx
                    kp1_coord = np.array(kp1[kp1_idx].pt, dtype=int)
                    kp2_coord = np.array(kp2[kp2_idx].pt, dtype=int)
                    cv2.line(frame, kp1_coord, kp2_coord, [0, 0, 255], 2)

            out.write(frame)
            kp1 = kp2
            des1 = des2
            
    out.release()
    print(f"complete: {path}")

# with Pool(2) as pool:
#     pool.map(job, path_list)

job(path)