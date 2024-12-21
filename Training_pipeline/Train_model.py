''' 
Fifth script: fine-tune the YOLO model.
Next step: Inference.py

NOTE: the scripts for the fine-tuning must be specified inside the 'if __name__ == '__main__':' strcuture.
'''

from ultralytics import YOLO
from pathlib import Path
import torch

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Object under exam
object = "Bottles"

if __name__ == '__main__':

    # Print some checks, if necessary
    print(f'{fonts.green}CUDA availability: {torch.cuda.is_available()}{fonts.reset}')
    print(f'{fonts.blue_light}The device for the training is: {torch.cuda.get_device_name(0)}{fonts.reset}')

    # Create an instance of the YOLO model and specify a custom directory
    model = YOLO("yolov8n.pt")
    output_dir = Path("Model")/object

    # In case of Windows 11 OS (GPU RTX 4070 ==> much bigger computational capability)    
    results = model.train(data = Path("Images")/object/"Labelled_set"/"data.yaml", 
                          epochs = 100, 
                          imgsz = 640, 
                          workers = 8, 
                          batch = 32, 
                          device = 0, 
                          lr0=1e-4, 
                          project = str(output_dir))