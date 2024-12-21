''' 
Fourth script: automatic labelling of the images.
Next step: Train_model.py
'''

import os
import random
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Define a variable for bounding boxes on the images in the training dataset
show_bounding_boxes = False

# Object under exam
object = "Bottles"
imposed_label = "bottle"

# Checks on input and output folders
input_folder = Path("Images")/object/"Before_labelling"
if not input_folder.exists():
    print(f'{fonts.red}Error: The input folder {input_folder} does not exist! The execution will be stopped. {fonts.reset}')
    sys.exit(1)

output_folder = Path("Images")/object/"Labelled_set"
output_folder.mkdir(parents = True, exist_ok = True)
print(f'{fonts.green}The output folder is: {output_folder} {fonts.reset}')

# Define the ratios and the colors for the training
split_ratios = {"train": 0.7, "valid": 0.2, "test": 0.1}
split_colors = {
    "train": (0, 255, 0),  # Green
    "valid": (0, 0, 255),  # Red
    "test": (255, 0, 0)    # Blue
}

# Create the sub-folders for the training dataset
for split in split_ratios.keys():
    os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, split, "labels"), exist_ok=True)

# Create the YOLO model
model = YOLO("yolov8n.pt")

# Get all images and shuffle them to randomize the splits
image_paths = list(Path(input_folder).glob("*.jpg"))
random.shuffle(image_paths)

# Calculate the split indices and split images
num_images = len(image_paths)
train_end = int(split_ratios["train"] * num_images)
valid_end = train_end + int(split_ratios["valid"] * num_images)
splits = {
    "train": image_paths[:train_end],
    "valid": image_paths[train_end:valid_end],
    "test": image_paths[valid_end:]
}

# Process and save the images
for split, paths in splits.items():
    for image_path in paths:

        # Read image using OpenCV
        img = cv2.imread(str(image_path))
        img_height, img_width = img.shape[:2]
        
        # Run YOLO detection => This is why the artificial dataset must contain only one object
        results = model(image_path)
        detections = results[0].boxes  # Access the first result's bounding boxes
        
        # Define output paths
        split_image_folder = os.path.join(output_folder, split, "images")
        split_label_folder = os.path.join(output_folder, split, "labels")
        output_image_path = os.path.join(split_image_folder, image_path.name)
        label_file = os.path.join(split_label_folder, f"{image_path.stem}.txt")
        
        if len(detections) > 0:  # If any objects are detected

            # Find the bounding box with the largest area
            largest_box = None
            largest_area = 0
            
            # In case many objects are detected, we take the one with the largest area
            for box in detections: 

                # Extract YOLO outputs
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
                area = (x_max - x_min) * (y_max - y_min)          
                if area > largest_area:
                    largest_area = area
                    largest_box = box

            if largest_box:
                # Extract coordinates for the largest box
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                confidence = largest_box.conf[0]
                class_id = int(largest_box.cls[0])
                
                # Normalize bounding box coordinates for YOLO format
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                if show_bounding_boxes:

                    # Draw the largest bounding box on the image
                    color = split_colors[split]  # Get the color for the current split
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                    # Optionally add label text
                    label = f"{class_id} {confidence:.2f}"
                    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Write to label file in YOLO format
                with open(label_file, "w") as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            # Save the image with the bounding box
            cv2.imwrite(output_image_path, img)

        else:  # If no objects are detected
            print(f'{fonts.red}Warning: No objects detected in {image_path.name}!{fonts.reset}')

# After processing, create the data.yaml file
data_file = output_folder / "data.yaml"
with open(data_file, "w") as f:
    f.write("names:\n")
    f.write(f"- {imposed_label}\n")
    f.write("nc: 1\n")
    f.write(f"test: { (output_folder / 'test' / 'images').resolve() }\n")
    f.write(f"train: { (output_folder / 'train' / 'images').resolve() }\n")
    f.write(f"val: { (output_folder / 'valid' / 'images').resolve() }\n")

print(f'{fonts.green}The labelling process has been completed!{fonts.reset}')
