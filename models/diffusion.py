import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from models.timembedding import TimeEmbedding
from unet import UNet, UNetOutputLayer

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutputLayer(320, 4) # Build the final output size, sam e as input image size of Unet, hear unet predict only noise
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8) Output of encoder
        # context: (Batch_Size, Seq_Len, Dim) Prompt from CLIP
        # time: (1, 320) time of noisification, position encoding practicly of information time

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output