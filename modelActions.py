import helpers.ProcessFeatures as ProcessFeatures
import helpers.ProcessImage as ProcessImage
from torchvision import models
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os


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



def create_target_image(target, model, steps, show_every, style_grams, style_weights, content_features, content_weight, style_weight, train, save_dir):
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
        train - Boolean for if we are training style grams
        save_dir - Checkpoint save name

        Outputs:
        This method will display the target image as its being created
    '''
    print("Creating the target image(s)...")
    # for displaying the target image, intermittently
    show_every = show_every

    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = steps  # decide how many iterations to update your image (5000)

    for ii in range(1, steps+1):
        
        # Get the features from your target image    
        # Then calculate the content loss
        target_features = ProcessFeatures.get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # Initialize the style loss to 0
        style_loss = 0
        # Iterate through each style layer and add to the style loss
        for layer in style_weights:
            # Get the "target" style representation for the layer
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

        # If we are training and are at the halfway point, save a "low" checkpoint
        if (ii == (steps/2)) and train:
            # Save the checkpoint
            save_checkpoint(save_dir + "-low.pth", model, style_grams, style_weights)


    if train:
        save_checkpoint(save_dir + "-high.pth", model, style_grams, style_weights)
    print("Done creating the target image(s).\n")


def save_checkpoint(save_path, model, style_grams, style_weights):
    '''
        Saves a checkpoint file for the style transfer model

        Inputs:
        save_path: The relative path, including the file name and extension, to save the checkpoint to
        model: The model that was used to train for the style_transfer
    '''
    print("\nSaving the checkpoint...")
    checkpoint = {'state_dict': model.state_dict(),
                  'style_grams': style_grams,
                  'style_weights': style_weights}
    
    # Check that the checkpoints directory exists. If not create one
    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    torch.save(checkpoint, "./checkpoints/" + save_path)
    print("Done saving the checkpoint.\n")



def load_checkpoint(load_path, gpu):
    '''
        Loads a checkpoint to be used for style transfer

        Inputs:
        save_path - The relative file path, including the file name and extension, of the checkpoint

        Output:
        returns the model loaded with the checkpoint
    '''
    print("Loading style gram checkpoint...")
    # Get the checkpoint from the file path
    checkpoint = torch.load( "./checkpoints/" + load_path)

    # Load the vgg19 model from torchvision
    model = getattr(models, 'vgg19')(pretrained = True).features

    # Load the state_dict from the checkpoint into the model
    model.load_state_dict(checkpoint['state_dict'])
    style_grams = checkpoint['style_grams']
    style_weights = checkpoint['style_weights']
    #model.content_features = checkpoint['content_features']

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Move the model to GPU, if available, and chosen
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)

    print("Done loading style gram checkpoint.\n")
    # Return the model
    return model, style_grams, style_weights
