# Detection and Segmentation
This repo is meant for Windows. All the instructions are given for this operating system.
## Setup
### Requirements
- keras
- tensorflow
- CUDA
- torch
### Virtual environment
Open a new terminal and type:
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

### Initial structure

### Training pipeline
For the fine-tuning of the YOLO model, you have to look at the scripts contained in the folder ```Training_pipeline ```, in the following order:
- Augmentation.py: one at a time all the images in ```Images\Beckers\Becker``` or ```Images\Bottles\Bottle``` ar rotated, translated and filled.

### Troubleshooting
If you experience any problem when installing packages, maybe it is beacuse you do not have the 'LongPathEnabled' option in your registry. To fix it, open your system registers (regedit) and navigate to
```
Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
```
At this point, go to the cell 'LongPathEnabled' and change the value from 0 to 1.
