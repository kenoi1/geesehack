from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n_p.pt")


# Perform object detection on an image
results = model("puddle3.png")
results[0].show()