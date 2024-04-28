import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

vidcap = cv2.VideoCapture("/media/belkanwar/SATA_CORE/lifts/data/micro travel short sample1.mp4")
ret, frame = vidcap.read()
h, w = frame.shape[:2]

edge = cv2.adaptiveThreshold(
    cv2.cvtColor(frame.astype(np.uint8)[int(h*1/4):int(h*3/4), int(w*1/4):int(w*3/4)], cv2.COLOR_BGR2GRAY), 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV, 21, 6).astype(bool)
edge_coord = np.where(edge)
X = np.concatenate(edge_coord, axis=1)

cluster = DBSCAN(eps=0.5, min_samples=2)
labels = cluster.fit_predict(X)