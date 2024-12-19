import shutil
from pathlib import Path

def copy_images_from_folders(input_folders, output_folder, extensions=None):
    if extensions is None:

        # Default set of common image extensions (adjust as needed)
        extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    output_folder = Path(output_folder)
    # Create output folder if it does not exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    for folder in input_folders:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"Warning: {folder_path} is not a directory. Skipping.")
            continue
        
        # Scan for images
        for file_path in folder_path.rglob("*"):
            if file_path.suffix.lower() in extensions and file_path.is_file():
                # Copy image to output folder
                destination = output_folder / file_path.name
                shutil.copy2(file_path, destination)
                print(f"Copied {file_path} to {destination}")

if __name__ == "__main__":

    # Specify if you need to work with bottles or with beckers
    object = "Beckers"

    # Example hardcoded input and output folders:
    input_folder_1 = Path("Images")/object/"Initial_images"
    input_folder_2 = Path("Images")/object/"Pasted_background"
    input_folder_3 = Path("Images")/object/"Temp_augmentation"
    output_folder = Path("Images")/object/"Before_labelling"

    # Adjoint vector of input folders
    input_folders = [input_folder_1, input_folder_2, input_folder_3]
    copy_images_from_folders(input_folders, output_folder)
