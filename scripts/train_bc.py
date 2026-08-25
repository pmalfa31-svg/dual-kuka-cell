import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class BimanualPolicy(nn.Module):
    """Rete neurale per controllo bimanuale end-to-end (Obs -> 14D Action)."""
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
            nn.Tanh()  # Output normalizzato in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("[*] Caricamento dataset...")
    data = np.load("data/robosuite_dual_dataset.npz")
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    obs_dim = obs.shape[1]
    action_dim = actions.shape[1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training su device: {device} | Obs Dim: {obs_dim} | Action Dim: {action_dim}")

    model = BimanualPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("[*] Avvio addestramento (30 epoche)...")
    for epoch in range(1, 31):
        total_loss = 0.0
        for batch_obs, batch_act in loader:
            batch_obs, batch_act = batch_obs.to(device), batch_act.to(device)
            
            optimizer.zero_grad()
            pred_act = model(batch_obs)
            loss = criterion(pred_act, batch_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_obs.size(0)

        epoch_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/30 | MSE Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), "data/bimanual_policy.pt")
    print("\n[+] Policy salvata con successo in data/bimanual_policy.pt")

if __name__ == "__main__":
    main()
