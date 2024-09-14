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

# Define the image loss functions
class ImageReconstructionLoss(nn.Module):
    def __init__(self):
        super(ImageReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_image, ground_truth_image):
        return self.mse_loss(predicted_image, ground_truth_image)

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).permute(0,1,2).numpy()  # Move channels to the last dimension and convert to numpy array


# Create a dataloader for the episodes - (action  action_ang  crop_info  depth_1.png  depth_2.png  flow.png  rgb_1.png  rgb_2.png), 
# And there are many episodes in the dataset inside a folder
# Methods should include get item and length and get images in numpy and tensor format for the episodes
# action, action_ang, crop_info are text files with a list of numbers in them - load them as numpy arrays
class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.episodes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])

        # Load the data for the episode
        action = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action')))
        action_ang = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action_ang')))
        crop_info = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'crop_info')))
        
        # Convert images to numpy arrays for visualization
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'))
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'))
        flow = load_image(os.path.join(episode_dir, 'flow.png'))
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))

        return {
            'action': action,
            'action_ang': action_ang,
            'crop_info': crop_info,
            'depth_1': depth_1,
            'depth_2': depth_2,
            'flow': flow,
            'rgb_1': rgb_1,
            'rgb_2': rgb_2
        }
    
    def get_images_numpy(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'))
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'))
        flow = load_image(os.path.join(episode_dir, 'flow.png'))
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))

        return depth_1, depth_2, flow, rgb_1, rgb_2
    
    def get_images_tensor(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        depth_1 = load_image(os.path.join(episode_dir, 'depth_1.png'))
        depth_2 = load_image(os.path.join(episode_dir, 'depth_2.png'))
        flow = load_image(os.path.join(episode_dir, 'flow.png'))
        rgb_1 = load_image(os.path.join(episode_dir, 'rgb_1.png'))
        rgb_2 = load_image(os.path.join(episode_dir, 'rgb_2.png'))
        # Return float32 

        return torch.tensor(depth_1).permute(0,1,2), torch.tensor(depth_2).permute(0,1,2), \
               torch.tensor(flow).permute(0,1,2), torch.tensor(rgb_1).permute(0,1,2), \
               torch.tensor(rgb_2).permute(0,1,2)
    
    
    def get_actions_numpy(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        action = np.loadtxt(os.path.join(episode_dir, 'action'))
        action_ang = np.loadtxt(os.path.join(episode_dir, 'action_ang'))
        crop_info = np.loadtxt(os.path.join(episode_dir, 'crop_info'))

        return action, action_ang, crop_info
    
    def get_actions_tensor(self, idx):
        episode_dir = os.path.join(self.root_dir, self.episodes[idx])
        action = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action')), dtype=torch.float32)
        action_ang = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'action_ang')) , dtype=torch.float32)
        crop_info = torch.tensor(np.loadtxt(os.path.join(episode_dir, 'crop_info')) , dtype=torch.float32)
        # Return float32 datatype only
        return action, action_ang, crop_info
    

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