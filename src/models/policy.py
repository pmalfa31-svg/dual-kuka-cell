import torch
import torch.nn as nn

class BimanualPolicy(nn.Module):
    """Architettura MLP con normalizzazione per controllo bimanuale a 14D."""
    def __init__(self, obs_dim: int, action_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
