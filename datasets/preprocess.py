"""
preprocess.py
=============

Preprocess raw videos for ISTVT.

Pipeline:
Video
    ↓
Frame Extraction
    ↓
MTCNN Face Detection
    ↓
Nose-centered Crop (1.25×)
    ↓
Resize to 300×300
    ↓
Save cropped face images
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mtcnn import MTCNN

from configs.config import (
    IMAGE_SIZE,
    FACE_CROP_SCALE,
    DEVICE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

def create_mtcnn():
    """
    Create an MTCNN face detector.
    """

    return MTCNN()

def extract_frames(video_path):
    """
    Read all frames from a video.

    Returns
    -------
    List[np.ndarray]
    """

    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    return frames

def crop_face(frame, mtcnn):
    """
    Detect the largest face and return a cropped face image.

    Parameters
    ----------
    frame : np.ndarray
        RGB image.

    mtcnn : MTCNN
        Initialized face detector.

    Returns
    -------
    PIL.Image or None
    """


    results = mtcnn.detect_faces(frame)

    if len(results) == 0:
        return None

    # --------------------------------------------------
    # Select largest detected face
    # --------------------------------------------------

    largest = max(
        results,
        key=lambda det: det["box"][2] * det["box"][3]
    )

    x, y, w, h = largest["box"]
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)

    face_width = w
    face_height = h

    # Nose landmark
    cx, cy = largest["keypoints"]["nose"]

    # Convert xywh → xyxy
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    # Paper: crop size = 1.25 × max(face width, face height)
    side = FACE_CROP_SCALE * max(face_width, face_height)
    half = side / 2

    left = int(round(cx - half))
    right = int(round(cx + half))
    top = int(round(cy - half))
    bottom = int(round(cy + half))

    img = frame

    h, w = img.shape[:2]

    # Reflection padding if crop exceeds image boundary
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - w)
    pad_bottom = max(0, bottom - h)

    if pad_left or pad_right or pad_top or pad_bottom:

        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top

    face = img[top:bottom, left:right]

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    return Image.fromarray(face)

def process_video(video_path, output_dir, mtcnn):
    """
    Process a single video.

    Parameters
    ----------
    video_path : Path
        Input video.

    output_dir : Path
        Folder where cropped faces are saved.

    mtcnn : MTCNN
        Face detector.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_frames(video_path)

    for idx, frame in enumerate(tqdm(frames, leave=False)):

        face = crop_face(frame, mtcnn)

        if face is None:
            continue

        save_path = output_dir / f"{idx:06d}.jpg"

        face.save(save_path)

def process_dataset():
    """
    Process every video under RAW_DATA_DIR.
    """

    mtcnn = create_mtcnn()

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    videos = []

    for ext in video_extensions:
        videos.extend(RAW_DATA_DIR.rglob(f"*{ext}"))

    print(f"Found {len(videos)} videos.")

    for video in tqdm(videos):

        relative = video.relative_to(RAW_DATA_DIR)

        output_dir = (
            PROCESSED_DATA_DIR /
            relative.parent /
            video.stem
        )

        process_video(
            video,
            output_dir,
            mtcnn,
        )

    print("Preprocessing complete.")

if __name__ == "__main__":
    process_dataset()