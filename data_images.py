import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm
import json

# ------------------------------
# STEP 1: Generate data.yaml
# ------------------------------

def generate_data_yaml(dataset_path):
    """
    Create a minimal data.yaml file for YOLO training.
    Updates the file to use 181 classes with specified class names.
    """
    class_names = [
        "3 Way Ball Valve",
        "3 Way Gate Valve",
        "3 Way Globe Valve",
        "4 Way Ball Valve",
        "4 Way Gate Valve",
        "Alkyalation",
        "Angle Blowdown",
        "Angle GlobeValve",
        "Angle Valve",
        "Automatic Stoker",
        "Axial Compressor",
        "Axical COMP",
        "BackPressure Regulator",
        "Balance Diaphragm Gate Valve",
        "Ball Valve",
        "Bleeder Valve",
        "Boom Loader",
        "Butterfly Valve",
        "Cavity Pump",
        "Centrifugal Blower",
        "Centrifugal Compressor",
        "Centrifugal Pump",
        "Check Valve",
        "Chimney Tower",
        "Closed Ball Valve",
        "Closed Gate Valve",
        "Cock Globe Valve",
        "Cock Valve",
        "Compressor Turbine",
        "Conveyor",
        "Counterflow Forced Draft",
        "Counterflow Natural Draft",
        "Crossflow Inducted",
        "Cylinder",
        "Diaphragm Valve",
        "Diesel Motor",
        "Double Flow Turbine",
        "Drum",
        "Ejector",
        "Electric Motor",
        "Electric operator gate valve",
        "Elevator",
        "Equipment ID",
        "Fail Closed Safe Position",
        "Fail Indeterminate Safe Position",
        "Fail Lock Safe Position",
        "Fail Open Safe Position",
        "Fan Blades",
        "Flanged",
        "Flanged Ball",
        "Flanged Butterfly",
        "Flanged Cock",
        "Flanged Ends",
        "Flanged Globe",
        "Float Operated Valve",
        "Flow Controller",
        "Flow Recorder",
        "Flow Transmitter",
        "Flowmeter",
        "Fluid Catalytic Cracking",
        "Fluid Cooking",
        "Fluidzed Recactor",
        "Furnace",
        "Gate Valve",
        "Gauge",
        "Gear Pump",
        "Globe Valve",
        "Hand Operated Angle Valve",
        "Hand Operated Ball Valve",
        "Hand Operated Gate Valve",
        "Hand Operated Globe Valve",
        "Heat Exchanger",
        "Hoist",
        "Horizontal Pump",
        "Horizontal Vessel",
        "Hydraulic Closed Valve",
        "Hydraulic Operated Ball Valve",
        "Hydraulic Operated Gate Valve",
        "Hydraulic Operated Globe Valve",
        "Hydrocracking",
        "Hydrodesulferization",
        "Integrated Block Valve",
        "Knife Valve",
        "Level Alarm",
        "Level Alarm High",
        "Level Alarm Low",
        "Level Controller",
        "Level Indicator",
        "Level Meter",
        "Level Recorder",
        "Level Transmitter",
        "Level Trasmitter",
        "Liquid Ring Vacuum",
        "Mangnetic",
        "Manhole",
        "Mixer",
        "Mixing Reactor",
        "Motor",
        "Motor Driven Turbine",
        "Motor Operated Angle Valve",
        "Motor Operated Butterfly Valve",
        "Motor Operated Diaphragm Valve",
        "Motor Operated Gate Valve",
        "Motor Operated Globe Valve",
        "Motor Operated Plug Valve",
        "Motor closed Gate Valve",
        "Needle Valve",
        "Not Gate",
        "Odometer",
        "Oil Burner",
        "Orifice",
        "Overhead Conveyor",
        "PRV",
        "PSV",
        "Packed Tower",
        "Pinch Valve",
        "Piston Operated Valve",
        "Plate Tower",
        "Plug Valve",
        "Pneumatic-Diaphragm 3 Way Valve",
        "Pneumatic-Diaphragm Angle Valve",
        "Pneumatic-Diaphragm Ball Valve",
        "Pneumatic-Diaphragm Butterfly Valve",
        "Pneumatic-Diaphragm Gate Valve",
        "Pneumatic-Diaphragm Globe Valve",
        "Positive Displacement",
        "Powered Gate Valve",
        "Pressure Balance Diaphragm",
        "Pressure Controller",
        "Pressure Gauge",
        "Pressure Indicating Controller",
        "Pressure Indicator",
        "Pressure Recorder",
        "Pressure Recording Controller",
        "Pressure Regulator",
        "Pressure Transmitter",
        "Progressive Cavity Pump",
        "Quarter Turn Valve Double Acting",
        "Quarter Turn Valve Spring Acting",
        "Reciprocation Compressor",
        "Relief Valve",
        "Rotameter",
        "Rotary Compressor",
        "Rotary Meter",
        "Rotary Piston-Pneumatic Butterfly Valve",
        "Rotary Piston-Pneumatic Closed Gate Valve",
        "Rotary Piston-Pneumatic Gate Valve",
        "Rotary Valve",
        "Sampler",
        "Scraper Conveyor",
        "Screw Conveyor",
        "Screw Pump",
        "Skip Hoist",
        "Slide Valve",
        "Socket Ends",
        "Socket Weld",
        "Solenoid Closed Valve",
        "Solenoid Operated Gate Valve",
        "Stop Valve",
        "Sump Pump",
        "Tank",
        "Temp Ind",
        "Temp Indicator",
        "Treaded",
        "Triple Fan Blade",
        "Tubular",
        "Turbine Driver",
        "Turbine Pump",
        "Vaccum Pump",
        "Vane Pump",
        "Vertical Pump",
        "Vertical Vessel",
        "Welded",
        "bolier",
        "pneumatic",
        "pressure safety Valve",
        "radio link",
        "reformer",
        "temp controller",
        "temp recorder",
        "temp trasmitter"
    ]
    
    data_yaml = {
        "path": dataset_path,
        "train": "train/images",
        "val": "valid/images",
        "nc": len(class_names),  # now set to 181
        "names": class_names
    }
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"✅ data.yaml created at {yaml_path}")

# ------------------------------
# STEP 2: Train YOLOv8 Model
# ------------------------------

def train_yolo(dataset_path, epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8 using the dataset specified by data.yaml.
    Returns the path to the best trained weights.
    """
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    model = YOLO("yolov8n.pt")  # Pre-trained YOLOv8 nano model
    model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz, batch=batch)
    print("✅ YOLO training complete!")
    # Best model is usually saved at runs/detect/train/weights/best.pt
    best_weights = os.path.join(dataset_path, "runs/detect/train/weights/best.pt")
    return best_weights

# ------------------------------
# STEP 3: Inference and Clustering on Test Images
# ------------------------------

def inference_and_cluster(dataset_path, trained_model_path, num_clusters=5):
    """
    Run inference on test images and cluster detected objects using KMeans.
    Clustering here uses the bounding box coordinates as features.
    """
    test_images_dir = os.path.join(dataset_path, "test/images")
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found at {test_images_dir}")
        return
    
    # Load trained YOLO model
    model = YOLO(trained_model_path)
    
    features = []  # to store bounding box features [x1, y1, width, height]
    detections_info = []  # for reference (image file and bbox)
    
    for img_file in tqdm(os.listdir(test_images_dir), desc="Inference on test images"):
        if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(test_images_dir, img_file)
            results = model(img_path)
            
            for result in results:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    width = x2 - x1
                    height = y2 - y1
                    features.append([x1, y1, width, height])
                    detections_info.append({
                        "image": img_file,
                        "bbox": [x1, y1, x2, y2]
                    })
    
    if len(features) == 0:
        print("❌ No detections found in test images.")
        return
    
    features = np.array(features)
    # Perform KMeans clustering on the bounding box features
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Assign cluster labels to detections_info
    for idx, det in enumerate(detections_info):
        det["cluster"] = int(cluster_labels[idx])
    
    # Save clustering results to JSON
    output_json = os.path.join(dataset_path, "cluster_results.json")
    with open(output_json, "w") as f:
        json.dump(detections_info, f, indent=4)
    
    print(f"✅ Clustering complete! Results saved to {output_json}")

# ------------------------------
# Main Execution
# ------------------------------

def main():
    dataset_path = "D:/dataset"
    
    # Step 1: Generate data.yaml if not exists
    generate_data_yaml(dataset_path)
    
    # Step 2: Train YOLOv8 model (this may be slow on CPU)
    best_weights = train_yolo(dataset_path, epochs=50, imgsz=640, batch=16)
    
    # Step 3: Run inference and cluster on test images
    inference_and_cluster(dataset_path, best_weights, num_clusters=5)

if __name__ == "__main__":
    main()
