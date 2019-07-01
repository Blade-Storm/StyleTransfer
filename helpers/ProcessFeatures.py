import torch

def get_features(image, model, layers=None):
    ''' 
        Run an image forward through a model and get the content features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)

        Imputs:
        image: The image to get the content features out of
        model: The model to use that extracts the content features from the convolutional layers
        layers: A dictionary with the index and name of each layer to extract features from 

        Outputs:
        Returns all of the content features 
    '''
    
    # Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',   # Style representation
                  '5': 'conv2_1',   # Style representation
                  '10': 'conv3_1',  # Style representation
                  '19': 'conv4_1',  # Style representation 
                  '21': 'conv4_2',  # Content representation
                  '28': 'conv5_1'}  # Style representation
    else:
        layers = layers
        
        
    features = {}
    x = image

    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    
    # Return the content features
    return features


def gram_matrix(tensor):
    ''' 
        Calculate the Gram Matrix of a given tensor that will represent the style of the image
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix

        Inputs:
        tensor: The image tensor to extract the style features from

        Outputs
        returns the gram matrix representing the style
    '''
    
    # Get the depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    # Reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, w*h)
    
    # Use matrix multiplication on the tensor and its transpose to get the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    # Return the gram matrix
    return gram 