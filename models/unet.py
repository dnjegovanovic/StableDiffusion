import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from models.attention import MultiHeadSelfAttention, MultiHeadCrossAttention


class ConditionalSequential(nn.Sequential):
    def forward(self, input_tensor, conditioning_context, timestep_embedding):
        """
        Processes the input tensor through the sequential layers of the model.

        Args:
            input_tensor (torch.Tensor): The input tensor with shape (Batch_Size, Channels, Height, Width).
            conditioning_context (torch.Tensor): The context tensor with shape (Batch_Size, Seq_Len, Dim).
            timestep_embedding (torch.Tensor): The time tensor with shape (1, Dim).

        Returns:
            torch.Tensor: The output tensor after processing through the layers.
        """
        for module in self:
            if isinstance(module, UNetAttentionBlock):
                input_tensor = module(input_tensor, conditioning_context)
            elif isinstance(module, UNetResidualBlock):
                input_tensor = module(input_tensor, timestep_embedding)
            else:
                input_tensor = module(input_tensor)
        return input_tensor


class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Upsamples the input tensor by a factor of 2 using nearest neighbor interpolation
        and applies a convolution operation.

        Args:
            x (torch.Tensor): The input tensor with shape (Batch_Size, Features, Height, Width).

        Returns:
            torch.Tensor: The output tensor with shape (Batch_Size, Features, Height * 2, Width * 2)
            after upsampling and convolution.
        """
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNetOutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x) -> torch.Tensor:
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class UNetResidualBlock(nn.Module):
    """_summary_
    Correlates the temporal embeddings with the UNet features.
    """

    def __init__(self, input_channels, output_channels, temporal_dim=1280):
        super().__init__()
        self.feature_groupnorm = nn.GroupNorm(32, input_channels)
        self.feature_conv = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, padding=1
        )
        self.temporal_linear = nn.Linear(temporal_dim, output_channels)

        self.merged_groupnorm = nn.GroupNorm(32, output_channels)
        self.merged_conv = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, padding=1
        )

        if input_channels == output_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(
                input_channels, output_channels, kernel_size=1, padding=0
            )

    def forward(self, feature_map, temporal_embedding):
        # feature_map: (Batch_Size, Input_Channels, Height, Width)
        # temporal_embedding: (1, Temporal_Dim)

        residual = feature_map

        # (Batch_Size, Input_Channels, Height, Width) -> (Batch_Size, Input_Channels, Height, Width)
        feature_map = self.feature_groupnorm(feature_map)

        # (Batch_Size, Input_Channels, Height, Width) -> (Batch_Size, Input_Channels, Height, Width)
        feature_map = F.silu(feature_map)

        # (Batch_Size, Input_Channels, Height, Width) -> (Batch_Size, Output_Channels, Height, Width)
        feature_map = self.feature_conv(feature_map)

        # (1, Temporal_Dim) -> (1, Temporal_Dim)
        temporal_embedding = F.silu(temporal_embedding)

        # (1, Temporal_Dim) -> (1, Output_Channels)
        temporal_embedding = self.temporal_linear(temporal_embedding)

        # Add width and height dimensions to temporal embedding.
        # (Batch_Size, Output_Channels, Height, Width) + (1, Output_Channels, 1, 1) -> (Batch_Size, Output_Channels, Height, Width)
        merged_output = feature_map + temporal_embedding.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Output_Channels, Height, Width) -> (Batch_Size, Output_Channels, Height, Width)
        merged_output = self.merged_groupnorm(merged_output)

        # (Batch_Size, Output_Channels, Height, Width) -> (Batch_Size, Output_Channels, Height, Width)
        merged_output = F.silu(merged_output)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged_output = self.merged_conv(merged_output)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged_output + self.residual_connection(residual)


class UNetAttentionBlock(nn.Module):
    """
    Implements a UNet-like attention block with self-attention, cross-attention,
    and feed-forward layers using GeGLU activation.
    """

    def __init__(self, num_heads: int, embedding_dim: int, context_dim=768):
        super().__init__()
        total_channels = num_heads * embedding_dim

        self.groupnorm = nn.GroupNorm(32, total_channels, eps=1e-6)
        self.input_projection = nn.Conv2d(
            total_channels, total_channels, kernel_size=1, padding=0
        )

        self.self_attn_norm = nn.LayerNorm(total_channels)
        self.self_attention = MultiHeadSelfAttention(
            num_heads, total_channels, input_proj_bias=False
        )

        self.cross_attn_norm = nn.LayerNorm(total_channels)
        self.cross_attention = MultiHeadCrossAttention(
            num_heads, total_channels, context_dim, input_proj_bias=False
        )

        self.ffn_norm = nn.LayerNorm(total_channels)
        self.geglu_linear_1 = nn.Linear(total_channels, 4 * total_channels * 2)
        self.geglu_linear_2 = nn.Linear(4 * total_channels, total_channels)

        self.output_projection = nn.Conv2d(
            total_channels, total_channels, kernel_size=1, padding=0
        )

    def forward(self, feature_map, context):
        """
        Forward pass for the attention block.

        Args:
            feature_map (torch.Tensor): Input feature map of shape (Batch_Size, Channels, Height, Width).
            context (torch.Tensor): Context tensor of shape (Batch_Size, Seq_Len, Dim).

        Returns:
            torch.Tensor: Processed feature map of the same shape as the input.
        """
        # feature_map: (Batch_Size, Channels, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        # Save initial input for final skip connection
        residual_input = feature_map

        # Group normalization and initial projection
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        feature_map = self.groupnorm(feature_map)

        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        feature_map = self.input_projection(feature_map)

        # Flatten spatial dimensions for attention processing
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height * Width)
        batch_size, channels, height, width = feature_map.shape
        feature_map = feature_map.view(batch_size, channels, height * width)

        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Height * Width, Channels)
        feature_map = feature_map.transpose(-1, -2)

        # Self-attention with residual connection
        # Save residual for skip connection
        # (Batch_Size, Height * Width, Channels)
        residual_short = feature_map

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.self_attn_norm(feature_map)

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.self_attention(feature_map)

        # Add skip connection: (Batch_Size, Height * Width, Channels)
        feature_map += residual_short

        # Cross-attention with residual connection
        # Save residual for skip connection
        # (Batch_Size, Height * Width, Channels)
        residual_short = feature_map

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.cross_attn_norm(feature_map)

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.cross_attention(feature_map, context)

        # Add skip connection: (Batch_Size, Height * Width, Channels)
        feature_map += residual_short

        # Feed-forward network with GeGLU and residual connection
        # Save residual for skip connection
        # (Batch_Size, Height * Width, Channels)
        residual_short = feature_map

        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.ffn_norm(feature_map)

        # GeGLU as implemented in the original code:
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Channels) -> two tensors of shape (Batch_Size, Height * Width, Channels * 4)
        feature_map, geglu_gate = self.geglu_linear_1(feature_map).chunk(2, dim=-1)

        # Element-wise product:
        # (Batch_Size, Height * Width, Channels * 4) * (Batch_Size, Height * Width, Channels * 4) -> (Batch_Size, Height * Width, Channels * 4)
        feature_map = feature_map * F.gelu(geglu_gate)

        # (Batch_Size, Height * Width, Channels * 4) -> (Batch_Size, Height * Width, Channels)
        feature_map = self.geglu_linear_2(feature_map)

        # Add skip connection: (Batch_Size, Height * Width, Channels)
        feature_map += residual_short

        # Reshape back to spatial dimensions
        # (Batch_Size, Height * Width, Channels) -> (Batch_Size, Channels, Height * Width)
        feature_map = feature_map.transpose(-1, -2)

        # (Batch_Size, Channels, Height * Width) -> (Batch_Size, Channels, Height, Width)
        feature_map = feature_map.view(batch_size, channels, height, width)

        # Final projection and skip connection between initial input and output of the block
        # (Batch_Size, Channels, Height, Width) + (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height, Width)
        return self.output_projection(feature_map) + residual_input


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
                ConditionalSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                ConditionalSequential(
                    UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                ConditionalSequential(
                    UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)
                ),
                # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
                ConditionalSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                ConditionalSequential(
                    UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                ConditionalSequential(
                    UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)
                ),
                # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
                ConditionalSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                ConditionalSequential(UNetResidualBlock(1280, 1280)),
                # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                ConditionalSequential(UNetResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = ConditionalSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNetResidualBlock(1280, 1280),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNetAttentionBlock(8, 160),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNetResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                ConditionalSequential(UNetResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
                ConditionalSequential(UNetResidualBlock(2560, 1280)),
                # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
                ConditionalSequential(UNetResidualBlock(2560, 1280), UpSample(1280)),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                ConditionalSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
                ConditionalSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
                ConditionalSequential(
                    UNetResidualBlock(1920, 1280),
                    UNetAttentionBlock(8, 160),
                    UpSample(1280),
                ),
                # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                ConditionalSequential(
                    UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)
                ),
                # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
                ConditionalSequential(
                    UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)
                ),
                # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(960, 640),
                    UNetAttentionBlock(8, 80),
                    UpSample(640),
                ),
                # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
                # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
                ConditionalSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
            ]
        )

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x
