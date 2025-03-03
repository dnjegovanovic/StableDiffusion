import torch
from stablediffusion.utils.timembedding_fun import get_batch_time_embedding


def test_timembedding():
    # Generate a batch of timesteps (e.g., during training)
    batch_size = 32
    timesteps = torch.randint(0, 1000, (batch_size,))  # Shape: [32]

    # Compute embeddings for the entire batch
    embeddings = get_batch_time_embedding(timesteps)  # Shape: [32, 320]
