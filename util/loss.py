import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define the image reconstruction loss
class ImageReconstructionLoss(nn.Module):
    def __init__(self):
        super(ImageReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_image, ground_truth_image):
        return self.mse_loss(predicted_image, ground_truth_image)
