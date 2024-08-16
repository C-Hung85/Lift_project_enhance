import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from scipy.sparse import lil_array
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
    move_pixel_coord_list = []
    
    while ret:
        ret, frame2 = vidcap.read()
        frame_idx += 1
        if ret and frame_idx % interval == 0:
            frame2 = frame2.astype(int)
            frame_diff = np.mean((frame2 - frame1)**2, axis=2, dtype=int)[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1]]
            move_pixel_coord = np.where((frame_diff > 120).reshape(-1))
            move_pixel_coord_list.append(move_pixel_coord)
        
    move_pixel_array = lil_array((len(move_pixel_coord_list), (h_roi[1]-h_roi[0])*(w_roi[1]-w_roi[0])), dtype=float)

    for idx, move_pixel_coord in enumerate(move_pixel_coord_list):
        move_pixel_array[idx, move_pixel_coord] = 1

    decomposer = NMF(n_components = N_DECOMPOSITION)
    weight = decomposer.fit_transform(move_pixel_array)
    feature = decomposer.components_

    for i in range(10):
        image = cv2.cvtColor(
            cv2.normalize(
                feature[i,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]),
                None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"images/{i}.png", image)