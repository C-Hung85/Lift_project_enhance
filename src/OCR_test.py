import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import easyocr
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

frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)

ocr_reader = easyocr.Reader(['en'], gpu=False)
ocr_reader.readtext(frame1, min_size=10, text_threshold=0.5, low_text=0.2)
