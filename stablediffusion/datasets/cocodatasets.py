import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoCaptions


class CocoCaptionDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            ann_file (string): Path to the COCO annotations file.
            transform (callable, optional): Optional transform to be applied on images.
            max_length (int): Maximum length for text captions.
        """
        self.coco_dataset = CocoCaptions(root=root_dir, annFile=ann_file)
        self.transform = transform

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        # Get image and all captions
        img, captions = self.coco_dataset[idx]

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        # Randomly select one caption
        caption = captions[torch.randint(0, len(captions), (1,)).item()]

        return img, caption
