import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import model_loader
from modules.pipeline import Pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch

import os

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

if __name__ == "__main__":
    
    # Hyperparameters
    hyperparameters = {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "epochs": 10,
        "num_inference_steps": 50,
        "diffusion_steps": 1000,
        "checkpoint_dir": './checkpoints',
    }

    os.makedirs(hyperparameters["checkpoint_dir"], exist_ok=True)
    
    tokenizer = CLIPTokenizer(
        "./data/tokenizer_vocab.json", merges_file="./data/tokenizer_merges.txt"
    )
    
    model_file = "./data/v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    # Define dataset and dataloader
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])



    dataset = CustomDataset("./data/train_images", transform=data_transforms, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    pipeline = Pipeline(
        models=models, tokenizer=tokenizer, device=DEVICE, idle_device="cpu"
    )