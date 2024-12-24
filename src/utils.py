import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cv2
import numpy as np
import numpy.typing as npt
from typing import Literal


def read_video(path:str, interval:int=1) -> dict:
    vidcap = cv2.VideoCapture(path)
    video = {'frames':[], 'fps':vidcap.get(cv2.CAP_PROP_FPS)}
    ret = True
    
    while ret:
        ret, frame = vidcap.read()

        if ret:
            video['frames'].append(frame)
    
    return video

def write_video(video:dict, filename:str) -> None:
    h, w = video['frames'][0].shape[:2]
    fps = video['fps']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join("data", "result", f"{filename}.mp4"), fourcc, fps, (w, h))
    [out.write(frame) for frame in video['frames']]
    out.release()

def remove_outlier_idx(input_array:npt.NDArray, mode:Literal['upper', 'lower', 'two-side']='upper'):
    q1 = np.quantile(input_array, 0.25)
    q3 = np.quantile(input_array, 0.75)
    
    if mode in {'upper', 'two-side'}:
        output_array = np.where(input_array <= q3 + 3.0*(q3-q1))[0]
    elif mode in {'lower', 'two-side'}:
        output_array = np.where(input_array >= q1 - 3.9*(q3-q1))[0]
    else:
        output_array = np.where((input_array <= q3 + 3.0*(q3-q1)) & (input_array >= q1 - 3.0*(q3-q1)))[0]

    return output_array