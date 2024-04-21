import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import numpy as np
from config import Config

interval = Config['scan_setting']['interval']

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts')):
    for file in files:
        path_list.append(os.path.join(root, file))

path = path_list[0]
vidcap = cv2.VideoCapture(path)
ret, frame1 = vidcap.read()
frame1 = frame1.astype(int)
h, w = frame1.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join("temp", "test.mp4"), fourcc, vidcap.get(cv2.CAP_PROP_FPS)/interval, (w, h))

counter = 0
while ret:
    ret, frame2 = vidcap.read()
    counter += 1
    if counter % interval == 0:
        frame2 = frame2.astype(int)
        frame_diff_raw = np.mean((frame2 - frame1)**2, axis=2, dtype=int)
        frame_diff_center = np.zeros_like(frame_diff_raw)
        frame_diff_center[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)] = frame_diff_raw[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)]
        # frame_diff = cv2.cvtColor(cv2.normalize(frame_diff_center,  None, 0, 255, cv2.NORM_MINMAX, dtype=8), cv2.COLOR_GRAY2BGR)

        frame_out = frame2.copy()
        frame_out[frame_diff_center>100] = [0, 255, 0]
        out.write(frame_out.astype(np.uint8))
        frame1 = frame2.copy()

out.release()
