import argparse
import torch
import modelActions
import helpers.ProcessImage as ProcessImage
import helpers.ProcessFeatures as ProcessFeatures

#######################################################
# Train a Neural Network using transfer learning to transfer the style from one image onto the content of another.
# This was taken from the Udacity: Deep Learning program: https://bit.ly/2FJqv8s
# Code for the program exersize: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/style-transfer
# 1. Get the directory to the content image
# 2. Get the directory for the checkpoint pth file
# 3. Get the name for the target image
# 4. Get the amount of steps (iterations) used for creating the target image
# 5. Get the amount of iterations to wait and show the progress of the target image
# 6. Choose GPU for training

# Create the parser and add the arguments
parser = argparse.ArgumentParser(description="Train a Neural Network using transfer learning")
# 1. Get the directory to the content image
parser.add_argument('content_directory', 
                    help="The relative path to the content image including the file name and extension.")
# 2. Get the directory for the checkpoint pth file
parser.add_argument('--checkpoint_directory',
                    help="The relative path to save the neural network checkpoint including the file name and extension")  
# 3. Get the name for the target image
parser.add_argument('--target_image', 
                    help="The name for target image. This will be used as the file name when saving.")   
# 4. Get the amount of steps (iterations) used for creating the target image
parser.add_argument('--steps', default=2000, type=int,
                    help="The amount of steps (iterations) used for creating the target image")   
# 5. Get the amount of iterations to wait and show the progress of the target image
parser.add_argument('--show_every', default=400, type=int,
                    help="The amount of iterations to wait and show the progress of the target image")         
# 6. Choose GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="If you would like to use the GPU for training. Default is False")

# Collect the arguments
args = parser.parse_args()
content_dir = args.content_directory
checkpoint_path = args.checkpoint_directory
target_image_name = args.target_image
steps = args.steps
show_every = args.show_every
gpu = args.gpu


# Load the vgg19 model from torchvision and set the state_dict from the checkpoint
model, style_grams, style_weights = modelActions.load_checkpoint(checkpoint_path, gpu)

# Load in content image
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
content = ProcessImage.load_image(content_dir).to(device)

# Get content and features only
content_features = ProcessFeatures.get_features(content, model)

# Create a third "target" image and prep it for change
# It is a good idea to start of with the target as a copy of our *content* image
# Then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# Set the alpha and beta for the content/style ratio. The larger the ratio (1/1) the less content ramains in the image
content_weight = 1  # alpha
style_weight = 1e6  # beta

# Create the target image
modelActions.create_target_image(target, model, steps, show_every, style_grams, style_weights, content_features, content_weight, style_weight)

# Save the target image
ProcessImage.save_target_image(target, target_image_name)