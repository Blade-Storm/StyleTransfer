import helpers.ProcessFeatures as ProcessFeatures
import helpers.ProcessImage as ProcessImage
from torchvision import models
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Get the "features" portion of VGG19 (we will not need the "classifier" portion)
model = models.vgg19(pretrained=True).features

# Freeze all VGG parameters since we're only optimizing the target image
for param in model.parameters():
    param.requires_grad_(False)

# Move the model to GPU, if available, and chosen
# TODO: Update this to include the argparse option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# TODO: Update this to use the argparse options
# Load in content and style image
content = ProcessImage.load_image('./Images/content/Building.jpg').to(device)
# Resize style to match content, makes code easier
style = ProcessImage.load_image('./Images/styles/Outrun.jpg', shape=content.shape[-2:]).to(device)

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(ProcessImage.im_convert(content))
ax2.imshow(ProcessImage.im_convert(style))


# Get content and style features only once before forming the target image
content_features = ProcessFeatures.get_features(content, model)
style_features = ProcessFeatures.get_features(style, model)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: ProcessFeatures.gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start of with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)


# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.8,
                 'conv3_1': 0.5,
                 'conv4_1': 0.3,
                 'conv5_1': 0.1}

# you may choose to leave these as is
content_weight = 1  # alpha
style_weight = 1e6  # beta


# TODO: Turn this into a function
# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
optimizer = optim.Adam([target], lr=0.003)
steps = 2000  # decide how many iterations to update your image (5000)

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
        
        ## TODO: Calculate the target gram matrix
        target_gram = ProcessFeatures.gram_matrix(target_feature)
        
        ## TODO:  get the "style" style representation
        style_gram = style_grams[layer]
        ## TODO: Calculate the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # add to the style loss
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


# display content and final, target image
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(ProcessImage.im_convert(content))
ax2.imshow(ProcessImage.im_convert(target))