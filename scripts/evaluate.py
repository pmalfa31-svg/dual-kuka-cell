import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import numpy as np
import robosuite as suite
from src.models.policy import BimanualPolicy

def main():
    print("[*] Avvio valutazione closed-loop...")
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=150,
        use_camera_obs=False
    )

    device = torch.device("cpu")
    model = BimanualPolicy(obs_dim=121, action_dim=14).to(device)
    model.load_state_dict(torch.load("data/bimanual_policy.pt", map_location=device))
    model.eval()

    for ep in range(1, 4):
        obs = env.reset()
        print(f"\n--- Episodio {ep}/3 ---")
        for step in range(150):
            flat_obs = np.concatenate([
                obs["robot0_proprio-state"],
                obs["robot1_proprio-state"],
                obs["object-state"]
            ], dtype=np.float32)

            with torch.no_grad():
                tensor_obs = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(tensor_obs).squeeze(0).cpu().numpy()

            obs, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.01)

            if done:
                print(f"[+] Task completato al frame {step}!")
                break

    env.close()

if __name__ == "__main__":
    main()
