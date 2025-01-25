from models.encoder import *
from models.decoder import *

WIDTH = 512
HEIGHT = 512


def test_decoder():
    input_shape = (1, 4, HEIGHT // 8, WIDTH // 8)
    input_tensor = torch.randn(input_shape)
    decoder = VAEDecoder()
    output_tensor = decoder(input_tensor)
    output_shape = (1, 3, HEIGHT, WIDTH)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_residual_block():
    input_shape = (1, WIDTH, HEIGHT, WIDTH)
    input_tensor = torch.randn(input_shape)
    resblock = VAEResidualBlock(HEIGHT, WIDTH)
    output_tensor = resblock(input_tensor)
    output_shape = (1, WIDTH, HEIGHT, WIDTH)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_attention_block():
    input_shape = (1, 64, HEIGHT // 4, WIDTH // 4)
    input_tensor = torch.randn(input_shape)
    attnblock = VAEAttentionBlock(64)
    output_tensor = attnblock(input_tensor)
    output_shape = (1, 64, HEIGHT // 4, WIDTH // 4)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"


def test_encoder_block():
    input_shape = (1, 3, HEIGHT, WIDTH)
    input_tensor = torch.randn(input_shape)
    noise_tensor = torch.randn(1, 4, HEIGHT // 8, WIDTH // 8)
    encoder = VAEncoder()
    output_tensor = encoder(input_tensor, noise_tensor)
    output_shape = (1, 4, HEIGHT // 8, WIDTH // 8)
    assert (
        output_tensor.shape == output_shape
    ), f"Expected output shape {output_shape}, got {output_tensor.shape}"
