import cv2
from ultralytics import YOLO

model = YOLO("yolo11n_p.pt")  # Use correct model name

cap = cv2.VideoCapture(0)  # Use 0 for webcam or video path for video file

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model.track(frame, persist=True)
        
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
            
            cv2.imshow("Live feed of something yay", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
