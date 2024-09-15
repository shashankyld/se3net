
# Create a dataloader for the episodes - (action  action_ang  crop_info  depth_1.png  depth_2.png  flow.png  rgb_1.png  rgb_2.png), 
# And there are many episodes in the dataset inside a folder
# Methods should include get item and length and get images in numpy and tensor format for the episodes
# action, action_ang, crop_info are text files with a list of numbers in them - load them as numpy arrays
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import numpy as np

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),  # Convert to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).permute(0,1,2).numpy()  # Move channels to the last dimension and convert to numpy array



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
    