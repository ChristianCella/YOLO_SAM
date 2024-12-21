''' 
Sixth script: test the fine-tuned model.
'''

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Object under exam
object = "Bottles"
label = "bottle"

model = YOLO(Path("Model")/object/"train/weights/best.pt")
image_path = Path("Images/Bottles/Validation_set/color_20241213_190319.png")
save_directory = Path("Images")/object/"Inference_results"
save_directory.mkdir(parents=True, exist_ok=True)

# Inference
results = model.predict(
    str(image_path), 
    imgsz = 640, 
    conf = 0.85, 
    save = False) 

# Result of the detection
detected_image = results[0].plot()

# Save the image
save_path = save_directory / f"{image_path.stem}_predictions.jpg"
plt.imsave(str(save_path), detected_image)

# Display the image
plt.imshow(detected_image)
plt.axis('off')  # Turn off axis for better visualization
plt.title(f"Detected {label}")
plt.show()
