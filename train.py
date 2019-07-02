import numpy as np
import torch
import torch.optim as optim
from torchvision import models
import modelActions
import argparse
import helpers.ProcessImage as ProcessImage
import helpers.ProcessFeatures as ProcessFeatures


#######################################################
# Train a Neural Network using transfer learning to transfer the style from one image onto the content of another.
# This was taken from the Udacity: Deep Learning program: https://bit.ly/2FJqv8s
# Code for the program exersize: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer
# 1. Get the directory to the content image
# 2. Get the directory to the style image
# 3. Set the directory to save the checkpoint with
# 4. Get a list of weights for the style
# 5. Get the amount of steps (iterations) used for creating the target image
# 6. Get the amount of iterations to wait and show the progress of the target image
# 7. Choose GPU for training

# Create the parser and add the arguments
parser = argparse.ArgumentParser(description="Train a Neural Network using transfer learning")
# 1. Get the directory to the content image
parser.add_argument('content_directory', 
                    help="The relative path to the content image including the file name and extension.")
# 2. Get the directory to the style image
parser.add_argument('style_directory', 
                    help="The relative path to the style image including the file name and extension.")
# 3. Set the directory to the image files to train with
parser.add_argument('--save_dir', default='./',
                    help="The relative path to save the neural network checkpoint")              
# 4. Get a list of weights for the style
parser.add_argument('--style_weights', default=[], nargs=5, type=float,
                    help="A list of weights to use on each convolutional layer for the style")
# 5. Get the amount of steps (iterations) used for creating the target image
parser.add_argument('--steps', default=2000, type=int,
                    help="The amount of steps (iterations) used for creating the target image")   
# 6. Get the amount of iterations to wait and show the progress of the target image
parser.add_argument('--show_every', default=400, type=int,
                    help="The amount of iterations to wait and show the progress of the target image")   
# 7. Choose GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="If you would like to use the GPU for training. Default is False")

# Collect the arguments
args = parser.parse_args()
content_dir = args.content_directory
style_dir = args.style_directory
save_dir = args.save_dir
style_weights_list = args.style_weights
steps = args.steps
show_every = args.show_every
gpu = args.gpu


# Create the model from VGG19
model = modelActions.create_model(gpu)

# Load in content and style images
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
content = ProcessImage.load_image(content_dir).to(device)
# Resize style to match content, makes code easier
style = ProcessImage.load_image(style_dir, shape=content.shape[-2:]).to(device)

# Get content and style features only once before forming the target image
content_features = ProcessFeatures.get_features(content, model)
style_features = ProcessFeatures.get_features(style, model)

# Calculate the gram matrices for each layer of our style representation
style_grams = {layer: ProcessFeatures.gram_matrix(style_features[layer]) for layer in style_features}

# Create a third "target" image and prep it for change
# It is a good idea to start of with the target as a copy of our *content* image
# Then iteratively change its style
target = content.clone().requires_grad_(True).to(device)


# Set the weights for each style layer 
# Weighting earlier layers more will result in *larger* style artifacts
# Excluding `conv4_2` our content representation
if style_weights_list == []:
    style_weights = {'conv1_1': 1.,
                     'conv2_1': 0.8,
                     'conv3_1': 0.5,
                     'conv4_1': 0.3,
                     'conv5_1': 0.1}
else:
    style_weights = {'conv1_1': style_weights_list[0],
                     'conv2_1': style_weights_list[1],
                     'conv3_1': style_weights_list[2],
                     'conv4_1': style_weights_list[3],
                     'conv5_1': style_weights_list[4]}

# Set the alpha and beta for the content/style ratio. The larger the ratio (1/1) the less content ramains in the image
content_weight = 1  # alpha
style_weight = 1e6  # beta

# Create the target image
modelActions.create_target_image(target, model, steps, show_every, style_grams, style_weights, content_features, content_weight, style_weight)


# Display the final target image with the original content image
ProcessImage.display_final_target(content, target)


# Save the checkpoint
modelActions.save_checkpoint(save_dir, model, style_grams, style_weights, content_features)