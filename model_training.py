from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n_p.pt")


# Resume training

# # Train the model
train_results = model.train(
    # resume=True,
    data="/home/dereklin/Documents/vscode/geesehack/datasets/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("/home/dereklin/Documents/vscode/geesehack/datasets/test/images/image1192_jpeg.rf.4e7af28cb8bdf241e46aa83c1b7d33e6.jpg")
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model