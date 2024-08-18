import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.sparse import lil_array
from multiprocessing import Pool
from config import Config

interval = Config['scan_setting']['interval']
display = True
N_DECOMPOSITION = 3

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
    
    frame_signal_dict = {field:[] for field in ['frame_idx', 'move_signal', 'block_signal', 'move_pixel_coord', 'move_pixel', 'edge']}
    h, w = frame1.shape[:2]
    h_roi, w_roi = [[int(h*1/4), int(h*3/4)], [int(w*1/4), int(w*3/4)]]

    while ret:
        ret, frame2 = vidcap.read()
        frame_idx += 1
        if ret and frame_idx % interval == 0:
            frame2 = frame2.astype(int)
            frame_diff = np.mean((frame2 - frame1)**2, axis=2, dtype=int)[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)]
            move_pixel = frame_diff > 120
            edge = cv2.adaptiveThreshold(
                cv2.cvtColor(frame2.astype(np.uint8)[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)], cv2.COLOR_BGR2GRAY), 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 6).astype(bool)

            move_pixel_coord = list(np.where(move_pixel))
            block_signal = np.count_nonzero(edge)
            edge[tuple(move_pixel_coord)] = False
            move_signal = np.count_nonzero(edge)
            
            frame_signal_dict['move_signal'].append(move_signal)
            frame_signal_dict['block_signal'].append(block_signal)
            frame_signal_dict['frame_idx'].append(frame_idx)
            # frame_signal_dict['move_pixel_coord'].append((move_pixel_coord[0]+int(h*1/4), move_pixel_coord[1]+int(w*1/4)))
            frame_signal_dict['move_pixel_coord'].append(np.where(move_pixel.reshape(-1))[0])
            frame_signal_dict['move_pixel'].append(move_pixel)
            frame_signal_dict['edge'].append(edge)
            frame1 = frame2.copy()
    
    q1, q3 = np.quantile(frame_signal_dict['move_signal'], 0.25), np.quantile(frame_signal_dict['move_signal'], 0.75)
    frame_signal_dict['move'] = np.zeros(len(frame_signal_dict['move_signal']), dtype=bool)
    frame_signal_dict['move'][np.where(np.array(frame_signal_dict['move_signal']) < q1-1.5*(q3-q1))] = True

    q1, q3 = np.quantile(frame_signal_dict['block_signal'], 0.25), np.quantile(frame_signal_dict['block_signal'], 0.75)
    frame_signal_dict['block'] = np.zeros(len(frame_signal_dict['block_signal']), dtype=bool)
    frame_signal_dict['block'][np.where(np.array(frame_signal_dict['block_signal']) < q1-5*(q3-q1))] = True

    frame_signal_dict = pd.DataFrame(frame_signal_dict)
    moving_frame_signal_dict = frame_signal_dict.loc[(frame_signal_dict['move']) & (frame_signal_dict['block']==False)]

    move_pixel_array = lil_array((moving_frame_signal_dict.shape[0], (h_roi[1]-h_roi[0])*(w_roi[1]-w_roi[0])), dtype=float)

    for idx, move_pixel_coord in enumerate(moving_frame_signal_dict['move_pixel_coord']):
        move_pixel_array[idx, move_pixel_coord] = 1

    decomposer = NMF(n_components = N_DECOMPOSITION)
    decomposer.fit(move_pixel_array)
    feature = decomposer.components_

    argmax_array = np.argmax(feature, axis=0).reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0])+1
    filter_array = np.zeros_like(argmax_array, dtype=bool)
    for channel in range(N_DECOMPOSITION):
        filter_array += feature[channel,:].reshape(h_roi[1]-h_roi[0], w_roi[1]-w_roi[0]) > 0.1
    argmax_array = argmax_array * filter_array
    

    combine_image = np.zeros((h_roi[1]-h_roi[0], w_roi[1]-w_roi[0], 3), dtype=np.uint8)
    
    for channel in range(N_DECOMPOSITION):
        channel_pixel_coord = np.where(argmax_array==channel+1)
        combine_image[channel_pixel_coord[0], channel_pixel_coord[1], channel] = 255

    cv2.imwrite("images/combine.png", combine_image)


    vidcap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, vidcap.get(cv2.CAP_PROP_FPS)/interval, (w, h))

    if display:
        frame_signal_dict['move_signal'] = np.max(frame_signal_dict['move_signal']) - np.array(frame_signal_dict['move_signal'])
        frame_signal_dict['block_signal'] = np.max(frame_signal_dict['block_signal']) - np.array(frame_signal_dict['block_signal'])
        max_move_signal = np.max(frame_signal_dict['move_signal'])
        max_block_signal = np.max(frame_signal_dict['block_signal'])

    for frame_idx, move_signal, move, block_signal, block, move_pixel_coord, move_pixel, edge in zip(
        frame_signal_dict['frame_idx'], 
        frame_signal_dict['move_signal'], 
        frame_signal_dict['move'],
        frame_signal_dict['block_signal'],
        frame_signal_dict['block'],
        frame_signal_dict['move_pixel_coord'],
        frame_signal_dict['move_pixel'],
        frame_signal_dict['edge']):

        if display:
            move_signal_display = round(move_signal/max_move_signal, 3)
            block_signal_display = round(block_signal/max_block_signal, 3)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = vidcap.read()

        if ret:
            for channel in range(N_DECOMPOSITION):
                display_pixel = (argmax_array==channel+1) * move_pixel
                frame[h_roi[0]:h_roi[1], w_roi[0]:w_roi[1], channel][np.where(display_pixel)] = 255

            if block:
                color = [0, 255, 255]
                condition = "blocked"
            else:
                if move:
                    color = [0, 255, 0]
                    condition = "moving"
                else:
                    color = [0, 0, 255]
                    condition = "static"
            
            cv2.putText(frame, f"move signal: {move_signal_display} | block signal: {block_signal_display}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, condition, (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(frame)

    out.release()
    print(f"complete: {path}")

with Pool(5) as pool:
    pool.map(job, path_list)
