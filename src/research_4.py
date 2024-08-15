import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import StandardScaler
from config import Config

interval = Config['scan_setting']['interval']
display = True
N_DECOMPOSITION = 10

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts','data')):
    for file in files:
        path_list.append(os.path.join(root, file))

path = "/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4"


def job(path):
    vidcap = cv2.VideoCapture(path)
    ret, frame1 = vidcap.read()
    frame_idx = 0
    frame1 = frame1.astype(int)
    
    h, w = frame1.shape[:2]
    h_roi, w_roi = [[int(h*1/4), int(h*3/4)], [int(w*1/4), int(w*3/4)]]
    frames_array = []

    while ret:
        ret, frame = vidcap.read()
        if ret:
            frame = np.mean(frame[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1]].astype(int), axis=2)
            frames_array.append(frame.reshape(-1))
    
    frames_array = np.array(frames_array)
    decomposer = TruncatedSVD(n_components = N_DECOMPOSITION)
    scaler = StandardScaler(with_std = False)
    frames_array = scaler.fit_transform(frames_array)
    weight = decomposer.fit_transform(frames_array)
    feature = decomposer.components_

    for i in range(10):
        image = cv2.cvtColor(
            cv2.normalize(
                feature[i,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]),
                None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"images/{i}.png", image)