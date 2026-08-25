import os
import sys

os.environ["MUJOCO_GL"] = "glfw"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import robosuite
import robosuite.macros as macros
import robosuite.utils.binding_utils as binding_utils

macros.SIMULATION_GL = "glfw"
binding_utils._MUJOCO_GL = "glfw"

import time
import torch
import numpy as np
import robosuite as suite
from src.models.vision_policy import VisuomotorPolicy

def main():
    print("[*] Avvio valutazione closed-loop della Visuomotor Policy...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=True,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=150,
        use_camera_obs=True
    )

    device = torch.device("cpu")
    # Calcolo dimensione dinamica della propriocezione dal reset
    obs = env.reset()
    sample_proprio = np.concatenate([obs["robot0_proprio-state"], obs["robot1_proprio-state"]])
    
    model = VisuomotorPolicy(proprio_dim=len(sample_proprio), action_dim=14).to(device)
    model.load_state_dict(torch.load("data/visuomotor_policy.pt", map_location=device))
    model.eval()

    num_episodes = 3
    print(f"[+] Policy caricata. Avvio {num_episodes} episodi di test autonomi guidati da telecamera...")

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        print(f"\n--- Episodio {ep}/{num_episodes} ---")

        for step in range(150):
            img = np.transpose(obs["agentview_image"], (2, 0, 1)).astype(np.float32) / 255.0
            proprio = np.concatenate([obs["robot0_proprio-state"], obs["robot1_proprio-state"]], dtype=np.float32)

            with torch.no_grad():
                tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
                tensor_prop = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(tensor_img, tensor_prop).squeeze(0).cpu().numpy()

            obs, _, done, _ = env.step(action)
            env.render()
            time.sleep(0.01)

            if done:
                print(f"[+] Task completato con successo al frame {step} via Vision!")
                break

    env.close()
    print("\n[+] Valutazione visiva completata.")

if __name__ == "__main__":
    main()
