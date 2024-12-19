from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from pathlib import Path
import sys

# Specify if you need to work with bottles or with beckers
object = "Beckers"

# Define how many 'synthetic' images should be generated per input image (for each image inside th folder)
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

# Define the input folder containing the images (and check if it exists)
input_folder = Path("Images")/object/"Initial_images"
if not input_folder.exists():
    print(f"Error: The input folder {input_folder} does not exist")
    sys.exit(1)

# Define the output folder for augmented images
output_folder = Path("Images/Beckers/Temp_augmentation")
output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the output folder exists, otherwise it is created

# Iterate over all image files in the input folder
for image_file in input_folder.glob("*.*"):  # Adjust the glob pattern if needed to match specific extensions (e.g., "*.jpg", "*.png")
    try:
        # Read the image
        img = io.imread(image_file)

        # Reshape the image for augmentation
        img = img.reshape((1,) + img.shape)

        # Perform the augmentation
        i = 0
        for batch in datagen.flow(
                img, 
                batch_size=16, 
                save_to_dir=str(output_folder), 
                save_prefix=image_file.stem + '_aug',  # Use the original filename as a prefix
                save_format='png'
        ):
            i += 1
            if i >= num_img:
                break

        print(f"Augmented images generated for {image_file.name}")
    except Exception as e:
        print(f"Error processing {image_file.name}: {e}")
