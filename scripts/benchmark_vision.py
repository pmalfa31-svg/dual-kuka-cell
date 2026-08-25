import os
import sys

os.environ["MUJOCO_GL"] = "glfw"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import robosuite
import robosuite.macros as macros
import robosuite.utils.binding_utils as binding_utils

macros.SIMULATION_GL = "glfw"
binding_utils._MUJOCO_GL = "glfw"

import torch
import numpy as np
import robosuite as suite
from tqdm import tqdm
from src.models.vision_policy import VisuomotorPolicy

def run_vision_benchmark(num_episodes: int = 20, horizon: int = 150):
    print(f"[*] Inizializzazione benchmark Visuomotor su {num_episodes} episodi...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=horizon,
        use_camera_obs=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = env.reset()
    sample_proprio = np.concatenate([obs["robot0_proprio-state"], obs["robot1_proprio-state"]])
    
    model = VisuomotorPolicy(proprio_dim=len(sample_proprio), action_dim=14).to(device)
    model.load_state_dict(torch.load("data/visuomotor_policy.pt", map_location=device))
    model.eval()

    successes = 0
    completion_steps = []
    pbar = tqdm(range(1, num_episodes + 1), desc="[Vision Benchmark]", unit="ep")

    for ep in pbar:
        obs = env.reset()
        ep_success = False

        for step in range(horizon):
            img = np.transpose(obs["agentview_image"], (2, 0, 1)).astype(np.float32) / 255.0
            proprio = np.concatenate([obs["robot0_proprio-state"], obs["robot1_proprio-state"]], dtype=np.float32)

            with torch.no_grad():
                tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
                tensor_prop = torch.tensor(proprio, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(tensor_img, tensor_prop).squeeze(0).cpu().numpy()

            obs, reward, done, _ = env.step(action)

            if done or reward > 0.9:
                ep_success = True
                completion_steps.append(step)
                break

        if ep_success:
            successes += 1

        pbar.set_postfix({"Success Rate": f"{(successes / ep) * 100:.1f}%"})

    env.close()

    print("\n" + "="*50)
    print(" RISULTATI BENCHMARK VISUOMOTOR POLICY (ConvNet)")
    print("="*50)
    print(f"Episodi Totali        : {num_episodes}")
    print(f"Successi              : {successes}/{num_episodes}")
    print(f"Success Rate          : {(successes / num_episodes) * 100:.2f}%")
    if completion_steps:
        print(f"Step Medi Completamento: {np.mean(completion_steps):.1f} frames")
    print("="*50)

if __name__ == "__main__":
    run_vision_benchmark()
