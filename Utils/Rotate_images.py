''' 
Useful script: create images containing objects oriented in specific ways.

Suggestion: run this script only if you obtain a heterogeneous dataset from Augmentation.py.
'''

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage.transform import rotate
from pathlib import Path

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Object under exam
object = "Bottles"

# Define specific rotations
specific_rotations = [0, 90, 180, 270]  # Specify the exact angles you want
num_img = len(specific_rotations)  # Number of synthetic images = number of rotations

# Define the augmentation transformations (excluding rotation here)
datagen = ImageDataGenerator(
    width_shift_range=0.02,
    height_shift_range=0.02,
    shear_range=0,
    zoom_range=0,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define a list of images to rotate
image_files = [
    Path("Images")/object/"Initial_images/20241203_185034.jpg",
    Path("Images")/object/"Initial_images/20241203_185235.jpg",
    Path("Images")/object/"Initial_images/20241203_185025.jpg"
]

# Define the output folder for augmented images
output_folder = Path("Images")/object/"Temp_augmentation"
output_folder.mkdir(parents=True, exist_ok=True)

# Iterate over all image files in the input folder
it = 0
for image_file in image_files:
    it = it + 1
    try:
        # Read the image
        img = io.imread(image_file)

        # Apply specific rotations
        for angle in specific_rotations:
            rotated_img = rotate(img, angle, resize=False, mode='edge')  # Rotate the image

            # Reshape for additional augmentations
            rotated_img = rotated_img.reshape((1,) + rotated_img.shape)

            # Apply other augmentations
            i = 0
            for batch in datagen.flow(
                    rotated_img,
                    batch_size=16,
                    save_to_dir=str(output_folder),
                    save_prefix=f"{image_file.stem}_rot{angle}",
                    save_format='png'
            ):
                i += 1
                print(f'{fonts.green}Generated children {i} for image number {it} {fonts.reset}')
                if i >= 1:  # Only generate one augmentation per rotation
                    break

        # Acknowledge the sucess
        print(f'{fonts.blue}Finished generating images for {image_file.name} {fonts.reset}')
    except Exception as e:
        print(f'{fonts.red}Error processing {image_file.name}: {e} {fonts.reset}')
