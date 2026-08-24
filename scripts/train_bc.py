import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import PPO

from src.envs.dual_arm_env import DualKukaPalletizeEnv

def main():
    data = np.load("data/expert_dataset.npz")
    obs_tensor = torch.tensor(data["obs"], dtype=torch.float32)
    act_tensor = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    env = DualKukaPalletizeEnv()
    os.makedirs("checkpoints", exist_ok=True)

    # Inizializza modello PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0
    )

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("[*] Addestramento Behavioral Cloning (Supervised Imitation)...")
    epochs = 40
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for b_obs, b_act in loader:
            optimizer.zero_grad()
            pred_act = model.policy.actor_bwd(b_obs) if hasattr(model.policy, "actor_bwd") else model.policy.forward(b_obs)[0]
            # Utilizzo distribuzione d'azione standard SB3
            dist = model.policy.get_distribution(b_obs)
            pred_mean = dist.distribution.mean
            loss = criterion(pred_mean, b_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  -> Epoca {epoch:2d}/{epochs} | MSE Loss: {avg_loss:.6f}")

    model.save("checkpoints/best_model")
    print("[+] Modello esperto salvato in checkpoints/best_model.zip")

if __name__ == "__main__":
    main()
