import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import cv2

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