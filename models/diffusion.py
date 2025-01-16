import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from models.timembedding import TimeEmbedding
from models.unet import UNet, UNetOutputLayer


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_embedding = TimeEmbedding(320)
        self.denoising_unet = UNet()
        self.output_layer = UNetOutputLayer(
            320, 4
        )  # Builds the final output size, same as the input image size of the UNet. Here, the UNet predicts only noise.

    def forward(self, latent_representation, conditioning_context, timestep):
        # latent_representation: (Batch_Size, 4, Height / 8, Width / 8) Output of encoder
        # conditioning_context: (Batch_Size, Seq_Len, Dim) Prompt from CLIP
        # timestep: (1, 320) Represents the time of noisification, effectively a positional encoding for the time information.

        # (1, 320) -> (1, 1280)
        timestep_embedding = self.temporal_embedding(timestep)
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        denoised_output = self.denoising_unet(
            latent_representation, conditioning_context, timestep_embedding
        )
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        final_output = self.output_layer(denoised_output)

        # (Batch, 4, Height / 8, Width / 8)
        return final_output
