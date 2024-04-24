import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import numpy as np
from multiprocessing import Pool
from config import Config

interval = Config['scan_setting']['interval']

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts','data')):
    for file in files:
        path_list.append(os.path.join(root, file))


def job(path):
    vidcap = cv2.VideoCapture(path)
    ret, frame1 = vidcap.read()
    frame_idx = 0
    frame1 = frame1.astype(int)
    
    frame_signal_dict = {'signal':[], 'frame_idx':[], 'move_pixel_coord':[]}
    h, w = frame1.shape[:2]

    while ret:
        ret, frame2 = vidcap.read()
        frame_idx += 1
        if ret and frame_idx % interval == 0:
            frame2 = frame2.astype(int)
            frame_diff = np.mean((frame2 - frame1)**2, axis=2, dtype=int)[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)]
            move_pixel = frame_diff > 120
            edge = cv2.adaptiveThreshold(
                cv2.cvtColor(frame2.astype(np.uint8), cv2.COLOR_BGR2GRAY), 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 6
                )[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)].astype(bool)

            move_pixel_coord = list(np.where(move_pixel))
            move_pixel_count = len(move_pixel_coord[0])
            edge[tuple(move_pixel_coord)] = False
            signal = np.count_nonzero(edge, axis=None)

            frame_signal_dict['signal'].append(signal)
            frame_signal_dict['frame_idx'].append(frame_idx)
            frame_signal_dict['move_pixel_coord'].append((move_pixel_coord[0]+int(h*1/4), move_pixel_coord[1]+int(w*1/4)))
            frame1 = frame2.copy()
    
    q1, q3 = np.quantile(frame_signal_dict['signal'], 0.25), np.quantile(frame_signal_dict['signal'], 0.75)
    frame_signal_dict['move'] = np.zeros(len(frame_signal_dict['signal']), dtype=bool)
    frame_signal_dict['move'][np.where(np.array(frame_signal_dict['signal'])<q1-1.5*(q3-q1))] = True

    vidcap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, vidcap.get(cv2.CAP_PROP_FPS)/interval, (w, h))

    for frame_idx, signal, move, move_pixel_coord in zip(
        frame_signal_dict['frame_idx'], 
        frame_signal_dict['signal'], 
        frame_signal_dict['move'],
        frame_signal_dict['move_pixel_coord']):

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = vidcap.read()

        frame[tuple(move_pixel_coord)] = [0, 255, 0]
        if move:
            cv2.putText(frame, f"pixel count: {signal}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2)
        else:
            cv2.putText(frame, f"pixel count: {signal}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)

        out.write(frame)


    out.release()

with Pool(5) as pool:
    pool.map(job, path_list)

# job("/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4")