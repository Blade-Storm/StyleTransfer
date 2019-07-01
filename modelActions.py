import helpers.ProcessFeatures as ProcessFeatures
import helpers.ProcessImage as ProcessImage
from torchvision import models
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


def create_model(gpu):
    '''
        Create the model from VGG19

        Input:
        gpu - Boolean to use the gpu or cpu

        Output
        returns the model
    '''
    print("Loading the model...")
    # Get the "features" portion of VGG19 (we will not need the "classifier" portion)
    model = models.vgg19(pretrained=True).features

    # Freeze all VGG parameters since we're only optimizing the target image
    for param in model.parameters():
        param.requires_grad_(False)

    # Move the model to GPU, if available, and chosen
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    print("Done loading the model.\n")
    # Return the model
    return model



def create_target_image(target, model, steps, show_every, style_grams, style_weights, content_features, content_weight, style_weight):
    '''
        Inputs:
        target - The target image to create. This should be a 'blank' image to start
        model - The model that will be used to extract the features of the content and style images
        steps - The amount of iterations to update the target image
        show_every - Displayes the target image as its being created, intermittently
        style_grams - The gram matrix for the styles. Each row represents a convolutional layer's style
        style_weights - The weights used to calculate the style loss
        content_features - The content features from the convolutional model
        content_weight - The ratio for the content weight to style weight. This, along with style_weight, controls how much content vs style is created on the target image.
        style_weight - The ratio for the style weight to content weight. This, along with content_weight, controls how much content vs style is created on the target image.

        Outputs:
        This method will display the target image as its being created
    '''
    print("Creating the target image...\n")
    # for displaying the target image, intermittently
    show_every = show_every

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = steps  # decide how many iterations to update your image (5000)

    for ii in range(1, steps+1):
        
        ## TODO: get the features from your target image    
        ## Then calculate the content loss
        target_features = ProcessFeatures.get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # iterate through each style layer and add to the style loss
        for layer in style_weights:
            # get the "target" style representation for the layer
            target_feature = target_features[layer]
            _, d, h, w = target_feature.shape
            
            # Calculate the target gram matrix
            target_gram = ProcessFeatures.gram_matrix(target_feature)
            
            # Get the "style" style representation
            style_gram = style_grams[layer]
            # Calculate the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            
            # Add to the style loss
            style_loss += layer_style_loss / (d * h * w)
            
            
        ## Calculate the *total* loss
        total_loss = (content_loss * content_weight) + (style_loss * style_weight)
        
        # Update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # display intermediate images and print the loss
        if  ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(ProcessImage.im_convert(target))
            plt.show()

    print("Done creating the target image.\n")


def save_checkpoint(save_path, model):
    '''
        Saves a checkpoint file for the style transfer model

        Inputs:
        save_path: The relative path, including the file name and extension, to save the checkpoint to
        model: The model that was used to train for the style_transfer
    '''
    print("Saving the checkpoint...\n")
    checkpoint = {'state_dict': model.state_dict()}

    torch.save(checkpoint, save_path)
    print("Done saving the checkpoint.\n")