import torch
from torch.autograd import Function
from torch.nn import Module
import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image

'''
   DepthImageToDense3DPoints(height,width,fy,fx,cy,cx) :
   DepthImageToDense3DPoints.forward(depth_img)
   DepthImageToDense3DPoints.backward(grad_output)

   DepthImageToDense3DPoints takes in a set of depth maps (B x 1 x N x M) and outputs a 3D point (B x 3 x N x M) for every pixel in the image:
	The 3D points are defined w.r.t a frame centered in the center of the image. The frame has the following convention:
		+ve X is increasing columns, +ve Y is increasing rows, +ve Z is moving closer to the objects in the scene
	X and Y co-ordinates are computed based on the passed in camera parameters using the following formula:
		X = (xp - cx)/fx 
		Y = (yp - cy)/fy
		where (fx,fy) are the focal length in (x,y) => (col,row) and
				(cx,cy) are the centers of projection in (x,y) => (col,row)
  	  	Z   = 0 is the image plane
	  	Z   > 0 goes forward from the image plane
	The parameters (fy,fx,cy,cx) are optional - they default to values for a 240 x 320 image from an Asus Xtion Pro
'''




class DepthImageToDense3DPointsFunction(Function):
    @staticmethod
    def forward(ctx, depth, height, width, base_grid, fy, fx, cy, cx):
        # Check dimensions (B x 1 x H x W)
        batch_size, num_channels, num_rows, num_cols = depth.size()
        assert(num_channels == 1)
        assert(num_rows == height)
        assert(num_cols == width)

        # Save context and parameters
        ctx.save_for_backward(depth)
        ctx.base_grid = base_grid
        ctx.height = height
        ctx.width = width

        # Compute output = (x, y, depth)
        output = depth.new_zeros((batch_size, 3, num_rows, num_cols))
        xy, z = output.narrow(1, 0, 2), output.narrow(1, 2, 1)
        z.copy_(depth)  # z = depth
        xy.copy_(depth.expand_as(xy) * ctx.base_grid.narrow(1, 0, 2).expand_as(xy))  # [x*z, y*z, z]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        depth, = ctx.saved_tensors
        base_grid = ctx.base_grid

        # Compute grad input
        grad_input = (base_grid.expand_as(grad_output) * grad_output).sum(1)

        return grad_input, None, None, None, None, None, None  # Return None for unused gradients

class DepthImageToDense3DPoints(Module):
    def __init__(self, height, width,
				 fy = 589.3664541825391 * 0.5,
				 fx = 589.3664541825391 * 0.5,
				 cy = 240.5 * 0.5,
				 cx = 320.5 * 0.5):
        super(DepthImageToDense3DPoints, self).__init__()
        self.height = height
        self.width = width
        self.base_grid = None
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def forward(self, input):
        if self.base_grid is None:
            self.base_grid = torch.ones(1, 3, self.height, self.width).type_as(input.data)  # (x,y,1)
            for j in range(self.width):  # +x is increasing columns
                self.base_grid[:, 0, :, j].fill_((j - self.cx) / self.fx)
            for i in range(self.height):  # +y is increasing rows
                self.base_grid[:, 1, i, :].fill_((i - self.cy) / self.fy)

        # Call the static forward method of the function
        return DepthImageToDense3DPointsFunction.apply(input, self.height, self.width, self.base_grid, self.fy, self.fx, self.cy, self.cx)



def load_depth_image(depth_path):
    """
    Load depth image as a numpy array.
    """
    depth_image = np.array(Image.open(depth_path)) / 1000.0  # Assuming depth is in millimeters, convert to meters
    return depth_image



# Function to load action data from action.txt
def load_action_data(action_path):
    with open(action_path, 'r') as file:
        actions = file.readlines()
    
    # Convert each line into a list of floats and flatten the list
    action_tensor = torch.tensor(
        [float(value) for action in actions for value in action.strip().split()],
        dtype=torch.float32
    )
    
    return action_tensor

# Function to load flow image
def load_flow_image(flow_path):
    flow_image = np.array(Image.open(flow_path))  # Load the flow image
    flow_tensor = torch.tensor(flow_image, dtype=torch.float32)  # Convert to tensor
    return flow_tensor

# Process a single episode folder to generate two point cloud tensors and additional data
def process_episode(episode_folder):
    depth_1_path = os.path.join(episode_folder, 'depth_1.png')
    depth_2_path = os.path.join(episode_folder, 'depth_2.png')
    action_path = os.path.join(episode_folder, 'action')
    flow_path = os.path.join(episode_folder, 'flow.png')

    # Load depth images
    depth_1 = load_depth_image(depth_1_path)
    depth_2 = load_depth_image(depth_2_path)

    # Load action data
    action_tensor = load_action_data(action_path)

    # Load flow image
    flow_tensor = load_flow_image(flow_path)

    # Convert depth images to tensors with the correct shape
    depth_1_tensor = torch.tensor(depth_1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    depth_2_tensor = torch.tensor(depth_2).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # Instantiate the DepthImageToDense3DPoints class
    height, width = depth_1_tensor.shape[2], depth_1_tensor.shape[3]
    depth_to_pointcloud = DepthImageToDense3DPoints(height=height, width=width)

    # Create point clouds for both depth images
    pointcloud1_tensor = depth_to_pointcloud(depth_1_tensor)
    pointcloud2_tensor = depth_to_pointcloud(depth_2_tensor)

    return pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor



def process_all_episodes(base_folder):
    """
    Process all episodes in the base folder.
    """
    episode_folders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]

    all_pointclouds = {}
    for episode_folder in episode_folders:
        episode_name = os.path.basename(episode_folder)
        print(f"Processing {episode_name}...")
        pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor = process_episode(episode_folder)
        all_pointclouds[episode_name] = (pointcloud1_tensor, pointcloud2_tensor, action_tensor, flow_tensor)
    # print(all_pointclouds)
    return all_pointclouds


