from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")


# Resume training

# # Train the model
train_results = model.train(
    # resume=True,
    data="/home/dereklin/Documents/vscode/geesehack/geesehack.v2i.yolov11/data.yaml",  # path to dataset YAML
    epochs=80,  # number of training epochs
    imgsz=640,  # training image size
    pretrained=False,
    save=True,
    save_period=5,
    workers=8,
    name="navigoose",
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/dereklin/Downloads/Screenshot_20250126_001036 (2).png")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model