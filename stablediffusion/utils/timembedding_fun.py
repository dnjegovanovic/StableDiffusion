import torch


def get_time_embedding(timestep):
    """
    Computes a time embedding for a given timestep using sinusoidal position encoding.

    Args:
        timestep (int or float): The timestep value to embed.

    Returns:
        torch.Tensor: A tensor representing the time embedding.
                      Shape: (1, 320), where 320 is twice the embedding size (160 * 2).

    Example:
        >>> embedding = get_time_embedding(10)
        >>> print(embedding.shape)
        torch.Size([1, 320])
    """
    embedding_dim = 160  # The dimensionality of the time embedding
    # Compute frequency terms based on the embedding dimensionality
    frequencies = torch.pow(
        10000,
        -torch.arange(start=0, end=embedding_dim, dtype=torch.float32) / embedding_dim,
    )

    # Compute the time-dependent sinusoidal embedding
    timestep_tensor = torch.tensor([timestep], dtype=torch.float32).unsqueeze(
        1
    )  # Shape: (1, 1)
    positional_values = timestep_tensor * frequencies.unsqueeze(0)  # Shape: (1, 160)

    # Concatenate sine and cosine components to create the embedding
    time_embedding = torch.cat(
        [torch.cos(positional_values), torch.sin(positional_values)], dim=-1
    )  # Shape: (1, 320)

    return time_embedding


def get_batch_time_embedding(timesteps, device):
    """
    Computes time embeddings for a batch of timesteps using sinusoidal position encoding.

    Args:
        timesteps (torch.Tensor): Tensor of timestep values to embed. Shape: (batch_size,).

    Returns:
        torch.Tensor: A tensor representing the time embeddings.
                      Shape: (batch_size, 320), where 320 is twice the embedding size (160 * 2).

    Example:
        >>> timesteps = torch.tensor([10, 20, 30])
        >>> embeddings = get_time_embedding(timesteps)
        >>> print(embeddings.shape)
        torch.Size([3, 320])
    """
    embedding_dim = 160  # The dimensionality of the time embedding
    # Compute frequency terms (shape: [1, embedding_dim])
    frequencies = torch.pow(
        10000,
        -torch.arange(start=0, end=embedding_dim, dtype=torch.float32, device=device)
        / embedding_dim,
    ).unsqueeze(
        0
    )  # Shape: [1, embedding_dim]

    # Compute positional values (shape: [batch_size, embedding_dim])
    timesteps = timesteps.float().unsqueeze(-1)  # Shape: [batch_size, 1]
    positional_values = (
        timesteps * frequencies
    )  # Broadcast to [batch_size, embedding_dim]

    # Concatenate sine and cosine components
    time_embedding = torch.cat(
        [torch.cos(positional_values), torch.sin(positional_values)], dim=-1
    )  # Shape: [batch_size, 2 * embedding_dim] -> [batch_size, 320]

    return time_embedding
