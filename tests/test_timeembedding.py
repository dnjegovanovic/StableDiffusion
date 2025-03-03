from stablediffusion.models.timembedding import TimeEmbedding
import torch


def test_timembedding():
    time_embedding = TimeEmbedding(320)
    input_tensor = torch.randn(1, 320)
    output_tensor = time_embedding(input_tensor)
    assert output_tensor.shape == (
        1,
        1280,
    ), f"Expected output shape (1, 1280), got {output_tensor.shape}"
