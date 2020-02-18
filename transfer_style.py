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
# 8. Set the name for the target image
# 9. Boolean for if we are training a new style gram matrix or using one

# Create the parser and add the arguments
parser = argparse.ArgumentParser(description="Train a Neural Network using transfer learning")
# 1. Get the directory to the content image: './Images/content/me.jpg'
parser.add_argument('--content_dir', required=True,
                    help="The relative path to the content image including the file name and extension.")
# 2. Get the directory to the style image: './Images/styles/starry-night.jpg'
parser.add_argument('--style_dir', 
                    help="The relative path to the style image including the file name and extension.")
# 3. Get or Set the name of the checkpoint file. Two will be created: name-low and name-high
parser.add_argument('--checkpoint_name', required = True,
                    help="The name of the checkpoint to get or set")              
# 4. Get a list of weights for the style
parser.add_argument('--style_weights', default=[], nargs=5, type=float,
                    help="A list of weights to use on each convolutional layer for the style")
# 5. Get the amount of steps (iterations) used for creating the target image
parser.add_argument('--steps', default=2000, type=int,
                    help="The amount of steps (iterations) used for creating the target image")   
# 6. Get the amount of iterations to wait and show the progress of the target image
parser.add_argument('--show_every', default=400, type=int,
                    help="The amount of iterations to wait and show the progress of the target image")   
# 7. Boolean to choose GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="A boolean flag for if you would like to use the GPU for training. Default is False")
# 8. Set the name for the target image: "starry-me"
parser.add_argument('--target_image', required=True, 
                    help="The name for target image. This will be used as the file name when saving.")  
# 9. Boolean for training a new style gram or using one
parser.add_argument('--train', default=False, action='store_true', 
                    help="A boolean flag for if we are training a new style gram matrix. Default is False")  

# Collect the arguments
args = parser.parse_args()
content_dir = args.content_dir
style_dir = args.style_dir
checkpoint_name = args.checkpoint_name
style_weights_list = args.style_weights
steps = args.steps
show_every = args.show_every
target_image_name = args.target_image
gpu = args.gpu
train = args.train

if train:
    # Create the model from VGG19
    model = modelActions.create_model(gpu)
else:
    # Load the vgg19 model from torchvision and set the state_dict from the checkpoint
    model, style_grams, style_weights = modelActions.load_checkpoint(checkpoint_name, gpu)

# Load in content and style images
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
content = ProcessImage.load_image(content_dir).to(device)

if train:
    # Resize style to match content, makes code easier
    style = ProcessImage.load_image(style_dir, shape=content.shape[-2:]).to(device)
    # Get style features
    style_features = ProcessFeatures.get_features(style, model)
    # Calculate the gram matrices for each layer of our style representation
    style_grams = {layer: ProcessFeatures.gram_matrix(style_features[layer]) for layer in style_features}


# Get content features only once before forming the target image
content_features = ProcessFeatures.get_features(content, model)
    
# Create a third "target" image and prep it for change
# It is a good idea to start of with the target as a copy of our *content* image
# Then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# Set the weights for each style layer if we are training.
# Weighting earlier layers more will result in *larger* style artifacts
# Excluding `conv4_2` our content representation
if train: 
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
style_weight = 1e20  # beta

# Create the target image
modelActions.create_target_image(target, model, steps, show_every, style_grams, style_weights, content_features, content_weight, style_weight, train, checkpoint_name)

# Display the final target image with the original content image
ProcessImage.display_final_target(content, target)

# Save the target image
ProcessImage.save_target_image(target, target_image_name)