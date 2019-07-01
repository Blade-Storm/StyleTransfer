import numpy as np
import torch
import torch.optim as optim
from torchvision import models

#######################################################
# Train a Neural Network using transfer learning to transfer the style from one image onto the content of another:
# 1. Get the directory to the content image
# 2. Get the directory to the style image
# 3. Get a list of weights for the style
# 4. Choose GPU for training

