import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pdb

# Define the image reconstruction loss
class ImageReconstructionLoss(nn.Module):
    def __init__(self):
        super(ImageReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_image, ground_truth_image):
        pdb.set_trace()
        print("Calling forward method for loss...")
        # Check for NaNs or Infs in input images
        if torch.isnan(predicted_image).any() or torch.isinf(predicted_image).any():
            raise ValueError("predicted_image contains NaN or Inf values.")
        if torch.isnan(ground_truth_image).any() or torch.isinf(ground_truth_image).any():
            raise ValueError("ground_truth_image contains NaN or Inf values.")
        
        # Optionally clamp the values to avoid extreme values
        predicted_image = torch.clamp(predicted_image, 0.0, 1.0)
        ground_truth_image = torch.clamp(ground_truth_image, 0.0, 1.0)
        print("Predicted Image: ", predicted_image)
        print("Ground Truth Image: ", ground_truth_image)
        
        return self.mse_loss(predicted_image, ground_truth_image)
