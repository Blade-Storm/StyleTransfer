from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path, max_size=400, shape=None):
    ''' 
        Load in and transform an image, making sure the image
        is <= 400 pixels in the x-y dims.

        Inputs:
        img_path: The relative page to the image including the image name and file extension
        max_size: The maximum size for the image to be. This is to reduce training time
        shape: The shape for the image to be. 

        Outputs:
        returns the image
    '''
    
    # Get the image and make sure its in RGB values
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing so get the image size to rezise it later if nessisary
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    # Resize the image and convert it to a normalized tensor
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # Discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    # Return the image
    return image


def im_convert(tensor):
    '''
        Convert a tensor to an image. 

        Inputs:
        tensor: The tensor to convert to an image from

        Outputs:
        return the un-normalized image
    '''
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def display_final_target(content, target):
    '''
        Displays the final target image with the original content image

        Inputs:
        content - The original content image
        target - The final target image
    '''
    # display content and final, target image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax2.imshow(im_convert(target))