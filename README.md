# FER-Emotion-Detection
CMPM-17 Final Project CNN 
All library needed:

Basic Model:

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 
from torch.optim.lr_scheduler import ReduceLROnPlateau

Confusion Matrix:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

Demo:

import cv2 (if use openCV)
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

Can either use Image demo (change the filename to the image you want to detect) or use openCV demo (will need a webcam)

This is a simple way of making a FER-2013/plus CNN detector. 