import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.vision_policy import VisuomotorPolicy
from src.data.vision_dataset import RoboticsVisionDataset

def main():
    print("[*] Caricamento dataset visivo...")
    dataset = RoboticsVisionDataset("data/robosuite_vision_dataset.npz")
    loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=False)

    proprio_dim = dataset.proprios.shape[1]
    action_dim = dataset.actions.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Training su {device} | Proprio Dim: {proprio_dim} | Action Dim: {action_dim}")

    model = VisuomotorPolicy(proprio_dim=proprio_dim, action_dim=action_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    num_epochs = 20
    print(f"[*] Avvio training ({num_epochs} epoche)...")

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        model.train()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{num_epochs:02d}", unit="batch", leave=False)
        for b_img, b_prop, b_act in pbar:
            b_img = b_img.to(device)
            b_prop = b_prop.to(device)
            b_act = b_act.to(device)

            optimizer.zero_grad()
            pred_act = model(b_img, b_prop)
            loss = criterion(pred_act, b_act)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b_img.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Avg Loss: {epoch_loss:.6f}")

    os.makedirs("data", exist_ok=True)
    out_path = "data/visuomotor_policy.pt"
    torch.save(model.state_dict(), out_path)
    print(f"\n[+] Policy Visuomotoria salvata in {out_path}")

if __name__ == "__main__":
    main()
