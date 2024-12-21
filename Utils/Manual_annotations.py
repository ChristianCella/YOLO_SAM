''' 
This script is a good alternative to Roboflow for manual annotations.
It allows you to draw bounding boxes around objects in images and save the annotations in YOLO format.
To use this script, you need to have a folder with all the images you want to annotate (Manual_annotations/Temp_input), 
    and another one to save the annotations (Manual_annotations/Temp_output).
By following the instructions displayed in the terminal, you can create the annotations for the images.

NOTE: draw the bounding box, then press 'n' and 's' (if the class is correct) to save the .txt file.
'''

import os
import cv2
from pathlib import Path

# Import the colors
import sys
sys.path.append(".")
import Utils.fonts as fonts

# Get the current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the folders
image_folder = Path("Utils/Manual_annotations/Temp_input")
label_folder = Path("Utils/Manual_annotations/Temp_output")

# Define the classes
classes_ids = ["becker", "bottle"]

# Initialize global variables
drawing = False
ix, iy = -1, -1
rectangles, classes, rectangles_save = [], [], []
current_class = 0
img = None 

# Mouse callback function to draw rectangles
def draw_rectangle(event, x, y, flags, params):
    global ix, iy, drawing, rectangles, classes, current_class

    if event == cv2.EVENT_LBUTTONDOWN:  # At the first click
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE: # While the mouse is moving
        if drawing:
            img_copy = img.copy()  # Make a copy to display while drawing
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2) # Draw the rectangle dynamically
            cv2.imshow("Image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:  # Once you release the mouse
        drawing = False

        # Add the rectangle coordinates and the current class to the lists
        rectangles.append((ix, iy, x, y))

        # Draw the final rectangle on the image
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Image", img)


# Function to save annotations in YOLO format
def save_annotations(image_filename, rectangles_save, classes):
    h, w, _ = img.shape
    label_filename = os.path.join(label_folder, os.path.splitext(os.path.basename(image_filename))[0] + '.txt')

    with open(label_filename, 'w') as f:
        for (x1, y1, x2, y2), class_id in zip(rectangles_save, classes):

            # Convert to YOLO format: class_id x_center y_center width height (normalized)
            x_center = (x1 + x2) / 2 / w
            y_center = (y1 + y2) / 2 / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Annotate images function
def annotate_images():
    global img, rectangles, classes, current_class

    # Get all image full file paths of the images inside the image folder
    image_files = [os.path.join(root, file)
                   for root, _, files in os.walk(image_folder)
                   for file in files if file.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for image_filename in image_files:

        # Load and resize the image
        img = cv2.imread(image_filename)
        h, w, _ = img.shape
        img = cv2.resize(img, (int(w/3),int(h/3)), interpolation=cv2.INTER_LINEAR)
    
        # Display the image
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", draw_rectangle) # 'draw_rectangle' function is called when a mouse event occurs

        # Reset rectangles and classes for the new image
        rectangles, rectangles_save = [], []
        classes = []

        # Instructions
        print(f'{fonts.red}Instructions:{fonts.reset}')
        print(f'{fonts.green}1. Use the mouse to draw bounding boxes around objects.{fonts.reset}')
        print(f'{fonts.green}2. Press \'c\' to switch between different classes.{fonts.reset}')
        print(f'{fonts.green}3. Press \'n\' to save the current annotation and move to the next.{fonts.reset}')
        print(f'{fonts.green}4. Press \'q\' to quit the annotation process.{fonts.reset}')
        print(f'{fonts.green}5. Press \'s\' to save all annotations for the current image.{fonts.reset}')

        while True:

            # Create a copy of the image to display the current class
            img_with_text = img.copy()
            cv2.putText(img_with_text, f"Class: {classes_ids[current_class]}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image", img_with_text)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Switch to the next class
                current_class = (current_class + 1) % len(classes_ids)
            elif key == ord('n'):  # Save current annotation and prepare for the next
                if len(rectangles) > 0:  # Ensure at least one rectangle was drawn
                    h, w, _ = img.shape
                    (x1, y1, x2, y2) = rectangles[-1]
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    print(f"{fonts.purple}Annotation: {rectangles[-1]} -> {x_center} {y_center} {width} {height} Class: {classes_ids[current_class]} -> {current_class} {fonts.reset}")
                    classes.append(current_class)  # Ensure correct class is appended
                    rectangles_save.append(rectangles[-1])
            elif key == ord('q'):  # Quit
                break
            elif key == ord('s'):  # Save annotations
                save_annotations(image_filename, rectangles_save, classes)
                print(f"{fonts.green}Annotations saved for {image_filename} {fonts.reset}")
                break

        # Close the window for the current image
        cv2.destroyAllWindows()


# Run the annotation function
if __name__ == '__main__':
    annotate_images()