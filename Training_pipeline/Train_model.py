from ultralytics import YOLO
from pathlib import Path
import torch

verbose = True

# Specify if you need to work with bottles or with beckers
object = "Beckers"

if __name__ == '__main__':

    # Print some checks, if necessary
    if verbose:
        print("CUDA available:", torch.cuda.is_available())
        print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    # Create an instance of the YOLO model and specify a custom directory
    model = YOLO("yolov8n.pt")
    output_dir = Path("Model")/object

    # In case of windows (GPU RTX 4070 ==> much bigger computational capability)    
    results = model.train(data = Path("Images")/object/"Labelled_dataset"/"data.yaml", epochs = 100, imgsz = 640, workers = 8, batch = 32, device = 0, lr0=1e-4, project = str(output_dir))