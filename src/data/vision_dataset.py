import numpy as np
import torch
from torch.utils.data import Dataset

class RoboticsVisionDataset(Dataset):
    """Dataset multimodale: RGB Image [3, 84, 84] + Proprioception -> 14D Actions."""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.images = torch.tensor(data["images"], dtype=torch.float32)
        self.proprios = torch.tensor(data["proprios"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.proprios[idx], self.actions[idx]
