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

path = path_list[2]
vidcap = cv2.VideoCapture(path)

ret, frame = vidcap.read()

frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

ocr_reader = easyocr.Reader(['en'], gpu=False)
text_result = ocr_reader.readtext(frame, mag_ratio=2, allowlist=[str(i) for i in range(10)], text_threshold=0.5, low_text=0.4, min_size=20)

for bonding_box, text, score in text_result:
    [x1, y1], [x2, y1], [x2, y2], [x1, y2] = bonding_box
    cv2.rectangle(frame, [int(x1), int(y1)], [int(x2), int(y2)], [0,255,0], 2)
    cv2.putText(frame, text, [int(x1), int(y1)], cv2.FONT_HERSHEY_SIMPLEX, 1, color=[0, 255, 0])

for bonding_box, text, score in text_result:
    [x1, y1], [x2, y1], [x2, y2], [x1, y2] = bonding_box
    print(text, [[x1, y1], [x2, y2]], score)
    

cv2.imwrite("temp/test.png", frame)