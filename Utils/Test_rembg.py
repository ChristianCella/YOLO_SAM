''' 
Use this script to test the library 'rembg' to remove the background from an image.

Link to the repo: https://github.com/danielgatis/rembg
'''

from rembg import remove
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

# Imports for latex plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Specify the path to the image and load it
image_path = Path("Images/Bottles/Initial_images/20241203_185034.jpg")
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
result = remove(image)

plt.figure(figsize = (10, 10))
plt.title(rf"\textbf{{Background removed}}", fontsize=20)
plt.imshow(result, cmap = 'viridis')
plt.axis('off')
plt.show()
