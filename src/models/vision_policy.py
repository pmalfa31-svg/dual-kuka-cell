import torch
import torch.nn as nn

class VisuomotorPolicy(nn.Module):
    """Architettura Visuomotoria leggera: ConvNet 3-Layer + Proprioception -> 14D Action."""
    def __init__(self, proprio_dim: int = 102, action_dim: int = 14, visual_feature_dim: int = 64):
        super().__init__()
        
        # Encoder Convoluzionale rapido ottimizzato per input 3x84x84
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [32, 42, 42]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [64, 21, 21]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # [64, 11, 11]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, visual_feature_dim),
            nn.LayerNorm(visual_feature_dim),
            nn.Mish()
        )
        
        # Policy Head (Multimodal Fusion)
        fusion_dim = visual_feature_dim + proprio_dim
        self.actor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Mish(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        visual_features = self.conv(rgb)
        x = torch.cat([visual_features, proprio], dim=-1)
        return self.actor(x)
