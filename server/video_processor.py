import cv2
import base64
import numpy as np
import torch

model = ""


def process_frame(frame_data: bytes) -> str:
    # implementation of object detection and return the processed frames
    # Decode the base64 frame
    np_frame: np.ndarray = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
    frame: np.ndarray = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)

    # implement object detection
    results = ""  # chatgpt suggests using yolo model: model(frame)
    detected_frame = np.ndarray = np.array(results.render()[0])

    _, buffer = cv2.imencode(".jpg", detected_frame)
    frame_base64: str = base64.b64encode(buffer).decode("utf-8")
    return frame_base64


def objectDetection():
    return ""
