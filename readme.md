# You (should) Only Look Once: YOLO 🕵️‍♂️
The code provided in this branch can be used for the fine-tuning of YOLO, whose original repository can be found here: https://github.com/ultralytics/ultralytics?tab=readme-ov-file. The icons used in this file are taken from this repo: https://github.com/ikatyang/emoji-cheat-sheet.

---
### **Requirements and OS**🪟🐍
The operating system used for the testing of the codes is Windows 11, since no integration with ROS was required. The interpreter used in the virtual environment (detailed in the section [Virtual environment](#environment)) was python 3.10: this is more than enough if you only need to perform the detection stage.

---

### **Core dependencies**
- ❇️ ultralytics
- ❇️ keras
- ❇️ tensorflow
- ❇️ CUDA
- ❇️ torch
- ❇️ rembg

---

### **Virtual environment** <a name="environment"></a>
Open a new terminal (advice: use the ones inside VScode, to avoid navigating to the desired directory) and type:
```
python -m venv venv
```
Remember to add the virtual environment in the .gitignore file. To activate the virtual environment, use the following command:
```
venv\Scripts\activate
```
To install all the dependencies, you can either run the command 
```
pip install -r requirements.txt
```
or you can install them manually. The second approach is preferred when you need to use CUDA, as in this case.

---

### Numpy, torch and torchvision
To check the CUDA version installed you can run the command
```
nvidia-smi
```
If you have no CUDA installed, you can access the following website and install the Toolkit:
https://developer.nvidia.com/cuda-downloads
CUDA needs a specific version of torch and torchvision. Look at the following webiste to insdtall the correct one:
https://pytorch.org/get-started/locally/; in my case, since I dispose of CUDA 12.4, I have to run the following command:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Now, if you run the code ```Test_version.py``` inside ```Utils```, you should see something like this:
```
PyTorch version: 2.5.1+cu124
Torchvision version: 0.20.1+cu124
CUDA available: True
CUDA version: 12.4
```

### Background removal
This repository relies on the package ```rembg``` to remove the abckground from the initial images and obtain new ones containing only the becker or the battle pasted on a new background. To be able to use the package, run the following commands in sequence:
```
pip install onnxruntime-gpu
pip install rembg
```

### Initial structure

---

### Training pipeline
For the fine-tuning of the YOLO model, you have to look at the scripts contained in the folder ```Training_pipeline```, in the following order:
- Augmentation.py: one at a time all the images in ```Images\...\Becker``` or ```Images\...\Bottle``` ar rotated, translated and filled. Important: after creating the dataset, look at it and remove all the images that are not meaningful.
- Create_new_dataset.py: the output of this script is saved in ```Images\...\Pasted_bacgrounds``` and it is the result of scanning all the augmented dataset, removing the background and pasting the object (either a becker or a bottle) on a new background.
- Create_training_dataset.py: This script is used to join the folders ```Initial_images```, ```Pasted_background```, ```Temp_auagmentation``` to create the folder ```Before_labelling``` that will be used for the labelling.
- Autoamtic_labelling.py: this script creates the folder ```Labelled_dataset``` with the file data.yaml automatically created.
- Train_model.py: This file takes as input the images in ```Labelled_dataset``` and trains the model for a specified number of epochs; the results are saved inside ```Model/.../train```

---

### **Troubleshooting**🎯
If you experience any problem when installing packages, maybe it is beacuse you do not have the 'LongPathEnabled' option in your registry. To fix it, open your system registers (regedit) and navigate to
```
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
```
At this point, go to the cell 'LongPathEnabled' and change the value from 0 to 1.
