import os
import yaml

# Paths
dataset_path = "D:/dataset"
labels_path = os.path.join(dataset_path, "labels")
yaml_path = os.path.join(dataset_path, "data.yaml")

# Ensure labels exist
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels folder not found at {labels_path}")

# Extract unique class names from labels
class_names = set()

for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)
    
    if os.path.isfile(label_path) and label_file.endswith(".txt"):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = parts[0]  # First value is the class ID
                    class_names.add(class_id)

# Sort class names
class_names = sorted(list(class_names))

# Generate YAML content
yaml_content = {
    "path": dataset_path,   # Root dataset directory
    "train": "images/",     # Train images
    "val": "images/",       # Using same images for validation
    "nc": len(class_names), # Number of unique classes
    "names": class_names    # List of class names
}

# Save YAML
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print(f"✅ `data.yaml` created successfully at {yaml_path}")
