import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import cv2
import warnings
import random
import utils
import datetime
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans
from multiprocessing import Pool
from config import Config, video_config

warnings.filterwarnings('ignore')

# Parameters and objects
DATA_FOLDER = Config['files']['data_folder']

for root, folder, files in os.walk(os.path.join(DATA_FOLDER, 'lifts','data')):
    for file in files:
        video_path = os.path.join(root, file)
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        ret, frame = vidcap.read()
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(video_config.get(file, {}).get('start', 0) * fps)
        end_frame = int(video_config.get(file, {}).get('end', video_length/fps) * fps)
        print(f"video: {file} | start: {int(start_frame/fps)} | end: {int(end_frame/fps)}")

        frame_idx_list = random.sample(range(start_frame, end_frame), 5)

        for idx, frame_idx in enumerate(frame_idx_list):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = vidcap.read()

            photo_name = f"{file.split(sep='.')[0]}-{idx}.png"
            cv2.imwrite(os.path.join(DATA_FOLDER, 'lifts', 'images', photo_name), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

