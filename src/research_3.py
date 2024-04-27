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
    
    frame_signal_dict = {field:[] for field in ['frame_idx', 'move_signal', 'block_signal', 'move_pixel_coord']}
    h, w = frame1.shape[:2]

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
            frame_signal_dict['move_pixel_coord'].append((move_pixel_coord[0]+int(h*1/4), move_pixel_coord[1]+int(w*1/4)))
            frame1 = frame2.copy()
    
    q1, q3 = np.quantile(frame_signal_dict['move_signal'], 0.25), np.quantile(frame_signal_dict['move_signal'], 0.75)
    frame_signal_dict['move'] = np.zeros(len(frame_signal_dict['move_signal']), dtype=bool)
    frame_signal_dict['move'][np.where(np.array(frame_signal_dict['move_signal']) < q1-1.5*(q3-q1))] = True

    q1, q3 = np.quantile(frame_signal_dict['block_signal'], 0.25), np.quantile(frame_signal_dict['block_signal'], 0.75)
    frame_signal_dict['block'] = np.zeros(len(frame_signal_dict['block_signal']), dtype=bool)
    frame_signal_dict['block'][np.where(np.array(frame_signal_dict['block_signal']) < q1-1.5*(q3-q1))] = True

    vidcap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path.replace("data", "result"), fourcc, vidcap.get(cv2.CAP_PROP_FPS)/interval, (w, h))

    for frame_idx, move_signal, move, block_signal, block, move_pixel_coord in zip(
        frame_signal_dict['frame_idx'], 
        frame_signal_dict['move_signal'], 
        frame_signal_dict['move'],
        frame_signal_dict['block_signal'],
        frame_signal_dict['block'],
        frame_signal_dict['move_pixel_coord']):

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = vidcap.read()

        if ret:
            frame[tuple(move_pixel_coord)] = [0, 255, 0]

            if block:
                color = [0, 255, 255]
            else:
                if move:
                    color = [0, 255, 0]
                else:
                    color = [0, 0, 255]
            cv2.putText(frame, f"move signal: {move_signal} | block signal: {block_signal}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(frame)

    out.release()

with Pool(5) as pool:
    pool.map(job, path_list)

# job("/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4")