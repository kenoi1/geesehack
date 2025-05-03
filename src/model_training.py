from ultralytics import YOLO

model = YOLO("yolo11s.pt")



train_results = model.train(
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

metrics = model.val()

results = model("/home/dereklin/Downloads/Screenshot_20250126_001036 (2).png")
results[0].show()

path = model.export(format="onnx")  # return path to exported model
