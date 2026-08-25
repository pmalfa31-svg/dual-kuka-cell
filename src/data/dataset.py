import numpy as np
import torch
from torch.utils.data import Dataset

class RoboticsDataset(Dataset):
    """Dataset loader per transizioni bimanuali Robosuite."""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.obs = torch.tensor(data["obs"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]
