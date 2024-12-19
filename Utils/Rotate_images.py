from keras.src.legacy.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage.transform import rotate
from pathlib import Path

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

# Define the input folder containing the images
input_folder = Path("Images/Bottles_to_rotate")
# Define the output folder for augmented images
output_folder = Path("Images/Rotated_bottles")
output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the output folder exists

# Iterate over all image files in the input folder
for image_file in input_folder.glob("*.*"):  # Adjust the glob pattern if needed to match specific extensions
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
                if i >= 1:  # Only generate one augmentation per rotation
                    break

        print(f"Augmented images generated for {image_file.name}")
    except Exception as e:
        print(f"Error processing {image_file.name}: {e}")
