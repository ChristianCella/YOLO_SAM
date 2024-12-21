''' 
Third script: create teh complete dataset for the labelling stage.
Next step: Autoamtic_labelling.py
'''

import shutil
from pathlib import Path

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

def copy_images_from_folders(input_folders, output_folder, extensions=None):

    if extensions is None:
        # Default set of common image extensions (adjust as needed)
        extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each input folder  
    for folder_path in input_folders:
        if not folder_path.is_dir():
            print(f'{fonts.red}Warning: {folder_path} is not a directory!{fonts.reset}.')
            continue
        
        # Scan the images
        for file_path in folder_path.rglob("*"):
            if file_path.suffix.lower() in extensions and file_path.is_file():

                # Copy image to output folder
                destination = output_folder / file_path.name
                shutil.copy2(file_path, destination)
                print(f'{fonts.green_light}Copied image to {destination} {fonts.reset}')


if __name__ == "__main__":

    # Specify if you need to work with bottles or with beckers
    object = "Bottles"

    # Specify all the directories
    input_folder_1 = Path("Images")/object/"Initial_images"
    input_folder_2 = Path("Images")/object/"Pasted_background"
    input_folder_3 = Path("Images")/object/"Temp_augmentation"
    output_folder = Path("Images")/object/"Before_labelling"

    # Adjoint vector of input folders
    input_folders = [input_folder_1, input_folder_2, input_folder_3]
    copy_images_from_folders(input_folders, output_folder)
