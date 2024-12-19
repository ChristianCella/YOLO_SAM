# This code allows to upload the model obtained with 'Train_model.py' to make inference

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

object = "Beckers"
label = "Becker"

model = YOLO(Path("Model")/object/"train/weights/best.pt")
image_path = Path("Images/Beckers/Becker_validation_set/color_20241213_190734.png")
save_directory = Path("Images")/object/"Inference_results"
save_directory.mkdir(parents=True, exist_ok=True) # Make sure the directory exists

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
