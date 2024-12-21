''' 
CLAHE (Contrast Limited Adaptive Histogram Equalization) is a technique used to enhance the contrast of an image.

NOTE: this is not used in the training pipeline, but it is a useful tool to enhance the images before segmenting them.
'''

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Imports for latex plots
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Load the image
input_rgb_path = str(Path("Images/Bottles/Initial_images/20241203_185257.jpg"))
image = cv2.imread(input_rgb_path)

# Pass from BGR to CIE LAB (L = Lightness, a = green-yellow, b = blue-yellow) color space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l_channel, a_channel, b_channel = cv2.split(lab)

# Apply CLAHE to the L-channel to enhance contrast (even further with the optional histogram equalization)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
l_channel = clahe.apply(l_channel)
l_channel = cv2.equalizeHist(l_channel)

# Merge the enhanced L-channel back with the original A and B channels
lab = cv2.merge((l_channel, a_channel, b_channel))

# Convert the LAB image back to the BGR color space
enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# (Optional) Further increase contrast using convertScaleAbs
alpha = 1.5  # Contrast control
beta = 2    # Brightness control
enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=alpha, beta=beta)

# (Optional) Apply Unsharp Masking for edge enhancement
blurred = cv2.GaussianBlur(enhanced_image, (0, 0), 3) # Smoothed image
enhanced_image = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0) # sharpened = original + α(original−blurred)

# Display the image
plt.figure(figsize = (10, 10))
plt.title(rf"\textbf{{Enhanced image}}", fontsize=20)
plt.imshow(enhanced_image, cmap = 'viridis')
plt.axis('off')
plt.show()
