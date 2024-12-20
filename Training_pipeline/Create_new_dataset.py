''' 
Second script: augment the number of images.
Next step: Create_traaining_dataset.py

NOTE: check the 'object' variable before starting and the threshold for the binary mask
(40 for beckers; 20 for bottles) 
'''

from PIL import Image, ImageChops
from rembg import remove
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import Counter
import sys

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Object under exam
object = "Bottles"

# Input directories
input_images_dir = Path("Images")/object/"Temp_augmentation"  # Folder containing all the images
backgrounds_dir = Path("Images/Background_images")  # Folder with the images of the backgrounds
if not input_images_dir.exists() or not backgrounds_dir.exists():
    print(f'{fonts.red}Error: One of the input directories does not exist {fonts.reset}')
    sys.exit(1)

# Output directories
output_dir = Path("Images")/object/"Pasted_background"
temp_dir = Path("Images")/object/"Isolated"

output_dir.mkdir(parents = True, exist_ok = True)
temp_dir.mkdir(parents = True, exist_ok = True)

# Load all the background images
background_files = glob(str(backgrounds_dir / "*.jpg"))
if not background_files:
    raise FileNotFoundError(f"{fonts.red}No background images found! {fonts.reset}")

# Load all the input images 
input_files = glob(str(input_images_dir / "*.png"))
if not input_files:
    raise FileNotFoundError(f"{fonts.red}No input images found! {fonts.reset}")

# Ensure balanced use of backgrounds
background_counter = Counter()
num_backgrounds = len(background_files)

# Process each input image
for idx, input_file in enumerate(input_files):

    # Load input image
    input_image = cv2.imread(input_file)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Remove background
    background_removed = remove(input_image)
    background_removed_rgb = cv2.cvtColor(background_removed, cv2.COLOR_BGR2RGB)
    
    # Save background-removed image temporarily
    temp_path = temp_dir / f"temp_{idx}.png"
    cv2.imwrite(str(temp_path), background_removed)

    # Load the processed image into PIL for further manipulation
    object_img = Image.open(temp_path).convert("RGB")

    # Create a binary mask
    mask = Image.eval(object_img.split()[2], lambda px: 0 if px < 20 else 255).convert("L")

    # Select a random background, ensuring balance
    selected_background = min(background_files, key=lambda bg: background_counter[bg])
    background_counter[selected_background] += 1

    # Load the selected background
    background_img = Image.open(selected_background).resize(object_img.size)

    # Composite the object onto the background
    output_img = Image.composite(object_img, background_img, mask)

    # Save the final composited image
    output_path = output_dir / f"output_{idx}.jpg"
    output_img.save(output_path)

    print(f"{fonts.cyan}Processed {idx + 1}/{len(input_files)}: Saved to {output_path} {fonts.reset}")

# Verify background distribution
print("Background usage count:", background_counter)
