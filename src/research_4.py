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
N_DECOMPOSITION = 3

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts','data')):
    for file in files:
        path_list.append(os.path.join(root, file))

path = "/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4"
os.system("rm images/*")


def job(path):
    vidcap = cv2.VideoCapture(path)
    ret, frame1 = vidcap.read()
    frame_idx = 0
    frame1 = frame1.astype(int)
    
    h, w = frame1.shape[:2]
    h_roi, w_roi = [[int(h*1/4), int(h*3/4)], [int(w*1/4), int(w*3/4)]]
    move_pixel_coord_list = []
    display_info = {'frame':[], 'edge':[], 'moving_pixel':[]}
    
    while ret:
        ret, frame2 = vidcap.read()
        frame_idx += 1
        if ret and frame_idx % interval == 0:
            display_info['frame'].append(frame2)
            
            frame2 = frame2.astype(int)
            frame_diff = np.mean((frame2 - frame1)**2, axis=2, dtype=int)[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1]]
            moving_pixel = frame_diff > 120
            move_pixel_coord = np.where(moving_pixel.reshape(-1))
            move_pixel_coord_list.append(move_pixel_coord)
            display_info['moving_pixel'].append(moving_pixel)

            edge = cv2.adaptiveThreshold(
                cv2.cvtColor(frame2.astype(np.uint8)[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1]], cv2.COLOR_BGR2GRAY), 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 6).astype(bool)
            display_info['edge'].append(edge)

            frame1 = frame2.copy()
            
        
    move_pixel_array = lil_array((len(move_pixel_coord_list), (h_roi[1]-h_roi[0])*(w_roi[1]-w_roi[0])), dtype=float)

    for idx, move_pixel_coord in enumerate(move_pixel_coord_list):
        move_pixel_array[idx, move_pixel_coord] = 1

    decomposer = NMF(n_components = N_DECOMPOSITION)
    weight = decomposer.fit_transform(move_pixel_array)
    feature = decomposer.components_

    combine_image = np.zeros((h_roi[1]-h_roi[0], w_roi[1]-w_roi[0], 3), dtype=np.uint8)


    for channel in range(N_DECOMPOSITION):
        image = 255*(feature[channel,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]) > 0.1).astype(np.uint8)
        combine_image[..., channel] = image
        cv2.imwrite(f"images/{channel}.png", cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

    cv2.imwrite("images/combine.png", combine_image)

    vidcap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, vidcap.get(cv2.CAP_PROP_FPS), (w, h))

    for frame, edge, moving_pixel in zip(display_info['frame'], display_info['edge'], display_info['moving_pixel']):
        for channel in range(3):
            channel_moving_pixel = moving_pixel * (feature[channel,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]) > 0.1)
            channel_edge = edge * (feature[channel,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]) > 0.1)
            frame[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1], channel][np.where(channel_edge)] = 255
            out.write(frame)
    
    out.release()
