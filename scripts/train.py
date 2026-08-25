import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.policy import BimanualPolicy
from src.data.dataset import RoboticsDataset

def main():
    print("[*] Caricamento dataset...")
    dataset = RoboticsDataset("data/robosuite_dual_dataset.npz")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.actions.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BimanualPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print(f"[*] Training su {device} (30 epoche)...")
    for epoch in range(1, 31):
        total_loss = 0.0
        for b_obs, b_act in loader:
            b_obs, b_act = b_obs.to(device), b_act.to(device)
            optimizer.zero_grad()
            pred = model(b_obs)
            loss = criterion(pred, b_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b_obs.size(0)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/30 | Loss: {total_loss / len(dataset):.6f}")

    os.makedirs("data", exist_ok=True)
    torch.save(model.state_dict(), "data/bimanual_policy.pt")
    print("[+] Policy salvata in data/bimanual_policy.pt")

if __name__ == "__main__":
    main()
