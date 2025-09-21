"""
影片旋轉校正工具函數模組
"""
import cv2
import numpy as np
import os


def rotate_frame(frame, angle_degrees):
    """
    剛性旋轉影像幀 (不進行縮放)
    
    Args:
        frame: 輸入影像
        angle_degrees: 旋轉角度 (逆時針為正值)
    
    Returns:
        rotated_frame: 旋轉後的影像 (維持原始尺寸，空白處黑色填充)
    """
    if angle_degrees == 0:
        return frame
    
    # 取得影像尺寸
    h, w = frame.shape[:2]
    
    # 取得影像中心點
    center = (w // 2, h // 2)
    
    # 建立旋轉矩陣 (縮放因子固定為 1.0，不進行縮放)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # 執行剛性旋轉變換
    rotated_frame = cv2.warpAffine(
        frame, 
        rotation_matrix, 
        (w, h),
        borderValue=(0, 0, 0)  # 黑色填充
    )
    
    return rotated_frame


def get_video_frames_for_preview(video_path, start_frame, end_frame, num_frames=2):
    """
    從影片中隨機取得指定數量的幀用於預覽
    
    Args:
        video_path: 影片路徑
        start_frame: 開始幀數
        end_frame: 結束幀數
        num_frames: 要取得的幀數 (預設為2)
    
    Returns:
        frames: 影像幀列表
    """
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    
    # 確保範圍有效
    if end_frame <= start_frame:
        video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = video_length
    
    # 隨機選擇幀數
    frame_indices = np.random.choice(
        range(start_frame, min(end_frame, int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)))), 
        size=min(num_frames, end_frame - start_frame), 
        replace=False
    )
    frame_indices = sorted(frame_indices)
    
    for frame_idx in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = vidcap.read()
        if ret:
            frames.append(frame)
    
    vidcap.release()
    return frames


def create_side_by_side_preview(frames, rotation_angle):
    """
    創建並排預覽圖片 (兩幀都經過旋轉)
    
    Args:
        frames: 影像幀列表
        rotation_angle: 旋轉角度
    
    Returns:
        preview_image: 並排預覽圖片
    """
    if len(frames) < 2:
        # 如果只有一幀，複製一份
        frames = frames + frames
    
    # 旋轉兩幀
    rotated_frame1 = rotate_frame(frames[0], rotation_angle)
    rotated_frame2 = rotate_frame(frames[1], rotation_angle)
    
    # 創建並排圖片
    h, w = rotated_frame1.shape[:2]
    side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # 放置兩幀
    side_by_side[:, :w] = rotated_frame1
    side_by_side[:, w:] = rotated_frame2
    
    # 加入文字標示
    cv2.putText(side_by_side, f"Frame 1 (Rotated {rotation_angle}°)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(side_by_side, f"Frame 2 (Rotated {rotation_angle}°)", 
                (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return side_by_side


def normalize_filename(filename):
    """
    標準化檔名，自動加入 .mp4 副檔名
    
    Args:
        filename: 使用者輸入的檔名
    
    Returns:
        normalized_filename: 標準化後的檔名
    """
    filename = filename.strip()
    if not filename:
        return filename
    
    # 如果沒有副檔名，自動加入 .mp4
    if not filename.endswith('.mp4'):
        filename += '.mp4'
    
    return filename


def validate_video_exists(data_folder, filename):
    """
    驗證影片檔案是否存在
    
    Args:
        data_folder: 資料資料夾路徑
        filename: 檔名
    
    Returns:
        exists: 檔案是否存在
        full_path: 完整檔案路徑
    """
    video_path = os.path.join(data_folder, 'lifts', 'data', filename)
    exists = os.path.exists(video_path)
    return exists, video_path
