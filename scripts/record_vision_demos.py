import os
import sys

# Backend grafico per Windows
os.environ["MUJOCO_GL"] = "glfw"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import robosuite
import robosuite.macros as macros
import robosuite.utils.binding_utils as binding_utils

# Patch runtime per rendering offscreen
macros.SIMULATION_GL = "glfw"
binding_utils._MUJOCO_GL = "glfw"

import numpy as np
import robosuite as suite
from tqdm import tqdm

def main():
    print("[*] Avvio registrazione dataset Visuomotor (RGB 84x84 + 14D Action)...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names="agentview",
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=100,
        use_camera_obs=True
    )

    images = []
    proprios = []
    actions = []
    num_episodes = 50

    pbar = tqdm(total=num_episodes, desc="[Vision Record]", unit="ep")

    for _ in range(num_episodes):
        obs = env.reset()
        ep_img, ep_prop, ep_act = [], [], []

        for step in range(100):
            # Normalizzazione RGB [0, 255] -> [0.0, 1.0] e layout [C, H, W]
            img = np.transpose(obs["agentview_image"], (2, 0, 1)).astype(np.float32) / 255.0
            proprio = np.concatenate([obs["robot0_proprio-state"], obs["robot1_proprio-state"]], dtype=np.float32)

            action = np.zeros(env.action_dim, dtype=np.float32)
            if step < 30:
                action[[0, 2, 6]] = [0.25, -0.30, -1.0]
                action[[7, 9, 13]] = [-0.25, -0.30, -1.0]
            elif step < 45:
                action[6] = 1.0
                action[13] = 1.0
            else:
                action[[2, 6]] = [0.40, 1.0]
                action[[9, 13]] = [0.40, 1.0]

            ep_img.append(img)
            ep_prop.append(proprio)
            ep_act.append(action)

            obs, _, _, _ = env.step(action)

        images.extend(ep_img)
        proprios.extend(ep_prop)
        actions.extend(ep_act)
        pbar.update(1)

    pbar.close()
    env.close()

    os.makedirs("data", exist_ok=True)
    out_path = "data/robosuite_vision_dataset.npz"
    np.savez_compressed(
        out_path,
        images=np.array(images, dtype=np.float32),
        proprios=np.array(proprios, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32)
    )
    print(f"\n[+] Dataset Visivo generato: {len(images)} frame salvati in {out_path}")

if __name__ == "__main__":
    main()
