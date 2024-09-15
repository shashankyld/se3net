import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from util.plot_utils import plot_image_from_se3_output, plot_image_from_se3_input_output_pair, plot_image_from_se3_input_output_gt
from models.se3net import SE3Net
from util.loss import ImageReconstructionLoss
from util.dataloader import EpisodeDataset


# Testing the dataset loader 
dataset = EpisodeDataset("/home/shashank/Documents/UniBonn/Sem4/alisha/Hind4Sight/Datasets/freiburg_real_poking/threeblocks/threeblocks/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


from matplotlib import pyplot as plt


'''
# To verify the dataset loader by index
for i in range(2):
    depth_1, depth_2, flow, rgb_1, rgb_2 = dataset.get_images_tensor(i)
    action, action_ang, crop_info = dataset.get_actions_tensor(i)
    # Print the shapes of the images and actions
    print(depth_1.shape, depth_2.shape, flow.shape, rgb_1.shape, rgb_2.shape)
    print(action.shape, action_ang.shape, crop_info.shape)
'''

# Load a batch of episodes and print the shapes of the images and actions
batch = next(iter(dataloader))
print(batch['depth_1'].shape, batch['depth_2'].shape, batch['flow'].shape, batch['rgb_1'].shape, batch['rgb_2'].shape)
print(batch['action'].shape, batch['action_ang'].shape, batch['crop_info'].shape)


model = SE3Net(3,4)
model.load_state_dict(torch.load("/home/shashank/Documents/UniBonn/Sem4/alisha/Hind4Sight/se3net/se3net_model.pth"))
x = batch['rgb_1']
x2 = batch['rgb_2']
u = batch['action'].float()
# Print datatypes of x, u
print(x.dtype, u.dtype)

# Forward pass
poses_new, x_new = model(x, u)
print(" x_new.shape, poses_new.shape ", x_new.shape, poses_new.shape)
plot_image_from_se3_output(x_new)
plot_image_from_se3_input_output_pair(x, x_new)
plot_image_from_se3_input_output_gt(x, x2, x_new)

# Check if the images have any nan or inf values, x, x2, x_new
if torch.isnan(x).any() or torch.isinf(x).any():
    raise ValueError("input_image contains NaN or Inf values.")
if torch.isnan(x_new).any() or torch.isinf(x_new).any():
    raise ValueError("predicted_image contains NaN or Inf values.")
if torch.isnan(x2).any() or torch.isinf(x2).any():
    raise ValueError("ground_truth_image contains NaN or Inf values.")


# Define the loss function
image_loss = ImageReconstructionLoss()
# Calculate the loss
image_loss_1 = image_loss(x_new, x2)
image_loss_2 = image_loss(x_new, x_new)
print("Image Loss: ", image_loss_1)
print("Image Loss 2: ", image_loss_2)

