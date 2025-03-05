from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the small YOLOv8 model

# Train YOLOv8
model.train(data="D:/dataset/data.yaml", epochs=5, imgsz=320, batch=16)

print("✅ YOLOv8 Training Complete!")

