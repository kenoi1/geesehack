import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open TCP stream
cap = cv2.VideoCapture('tcp://192.168.233.149:8000')

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Run YOLO detection
            results = model.track(frame, persist=True)
            
            if results and len(results) > 0:
                # Draw detections
                annotated_frame = results[0].plot()
                
                # Show frame
                cv2.imshow('YOLO Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to receive frame")
            break

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()