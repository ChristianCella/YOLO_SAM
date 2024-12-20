''' 
First script: augment the number of images.
Next step: Create_new_dataset.py

Suggestion: check images yourself before proceeding to the next step. If you see that some orientations are missing,
go to the script Rotate_images.py before proceeding.

NOTE: check the 'object' variable before starting! 
'''

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from pathlib import Path
import os

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Object under exam
object = "Bottles"

# Number of children images to generate for each image
num_img = 10

# Define the augmentation transformations
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directories
input_folder = Path("Images")/object/"Initial_images"
if not input_folder.exists():
    print(f"Error: The input folder {input_folder} does not exist")
    sys.exit(1)

output_folder = Path("Images")/object/"Temp_augmentation"
output_folder.mkdir(parents=True, exist_ok=True)

# Iterate over all image files in the input folder
it = 0
for image_file in input_folder.glob("*.*"):
    it = it + 1
    try:
        # Read the image and reshape it for augmentation
        img = io.imread(image_file)
        img = img.reshape((1,) + img.shape)

        # Augmentation
        i = 0
        for batch in datagen.flow(
                img, 
                batch_size=16, 
                save_to_dir=str(output_folder), 
                save_prefix=image_file.stem + '_aug',
                save_format='png'
        ):
            # Increment the counter
            i += 1
            print(f'{fonts.green}Generated children {i} for image number {it} {fonts.reset}')
            if i >= num_img:
                break
        # Acknowledge the sucess
        print(f'{fonts.blue}Finished generating images for {image_file.name} {fonts.reset}')
    except Exception as e:
        print(f'{fonts.red}Error processing {image_file.name}: {e} {fonts.reset}')
