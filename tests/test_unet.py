from stablediffusion.models.unet import *
from conftest import *

import pytest


def test_upsample():
    input_shape = (1, 3, HEIGHT // 8, WIDTH // 8)
    input_tensor = torch.randn(input_shape)
    upsample_layer = UpSample(3)
    output_tensor = upsample_layer(input_tensor)
    output_shape = (1, 3, HEIGHT // 8 * 2, WIDTH // 8 * 2)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_unet_output():
    input_shape = (1, 320, HEIGHT // 8, WIDTH // 8)
    input_tensor = torch.randn(input_shape)
    unet_output = UNetOutputLayer(320, 4)
    output_tensor = unet_output(input_tensor)
    output_shape = (1, 4, HEIGHT // 8, WIDTH // 8)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_unet_residual_block():
    feature_map = (1, 320, HEIGHT, WIDTH)
    feature_map = torch.randn(feature_map)
    temporal_embedding = (1, 1280)
    temporal_embedding = torch.randn(temporal_embedding)
    unet_res_block = UNetResidualBlock(320, 320)
    output_tensor = unet_res_block(feature_map, temporal_embedding)
    output_shape = (1, 320, HEIGHT, WIDTH)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_unet_attention_block():
    feature_map = (1, 320, HEIGHT // 8, WIDTH // 8)
    feature_map = torch.randn(feature_map)
    context = (1, 77, 768)
    context = torch.randn(context)
    unet_attn_block = UNetAttentionBlock(8, 40)
    output_tensor = unet_attn_block(feature_map, context)
    output_shape = (1, 320, HEIGHT // 8, WIDTH // 8)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_unet():
    input_shape = (1, 4, HEIGHT // 8, WIDTH // 8)
    input_tensor = torch.randn(input_shape)
    context = (1, 77, 768)
    context = torch.randn(context)
    time = torch.randn(1280, dtype=torch.float32)
    unet = UNet()
    output_tensor = unet(input_tensor, context, time)
    output_shape = (1, 320, HEIGHT // 8, WIDTH // 8)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"
