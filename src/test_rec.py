from ultralytics import YOLO

model = YOLO("yolo11n_p.pt")


results = model("puddle3.png")
results[0].show()
