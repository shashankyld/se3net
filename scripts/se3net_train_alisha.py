import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from util.plot_utils import plot_image_from_se3_output, plot_image_from_se3_input_output_pair, plot_image_from_se3_input_output_gt
from models.se3net import SE3Net
from util.loss import ImageReconstructionLoss
from util.dataloader import EpisodeDataset
from torchvision import transforms

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).permute(0,1,2).numpy()  # Move channels to the last dimension and convert to numpy array



# Testing the dataset loader 
dataset = EpisodeDataset("/home/shashank/Documents/UniBonn/Sem4/alisha/Hind4Sight/Datasets/freiburg_real_poking/threeblocks/threeblocks/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)



# Load a batch of episodes and print the shapes of the images and actions
batch = next(iter(dataloader))

model = SE3Net(3,4)
