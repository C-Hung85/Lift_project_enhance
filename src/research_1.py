import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import numpy as np
from config import Config

path_list = []
for root, folder, files in os.walk(os.path.join(Config['files']['data_folder'], 'lifts')):
    for file in files:
        path_list.append(os.path.join(root, file))

path = path_list[0]
vidcap = cv2.VideoCapture(path)

ret, frame1 = vidcap.read()
for i in range(12):
    ret, frame2 = vidcap.read()

frame1 = frame1.astype(int)
frame2 = frame2.astype(int)

frame_diff = cv2.normalize(np.mean(frame2 - frame1, axis=2),  None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("temp/test.png", frame_diff)