import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import robosuite as suite
from tqdm import tqdm

def main():
    print("[*] Generazione dataset bimanuale...")
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=100,
        use_camera_obs=False
    )

    observations, actions = [], []
    num_episodes = 50
    pbar = tqdm(total=num_episodes, desc="[Record]", unit="ep")

    for _ in range(num_episodes):
        obs = env.reset()
        ep_obs, ep_act = [], []
        
        flat_obs = np.concatenate([
            obs["robot0_proprio-state"],
            obs["robot1_proprio-state"],
            obs["object-state"]
        ], dtype=np.float32)

        for step in range(100):
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

            ep_obs.append(flat_obs)
            ep_act.append(action)

            obs, _, _, _ = env.step(action)
            flat_obs = np.concatenate([
                obs["robot0_proprio-state"],
                obs["robot1_proprio-state"],
                obs["object-state"]
            ], dtype=np.float32)

        observations.extend(ep_obs)
        actions.extend(ep_act)
        pbar.update(1)

    pbar.close()
    env.close()

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/robosuite_dual_dataset.npz",
        obs=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32)
    )
    print(f"[+] Dataset salvato con successo ({len(observations)} transizioni).")

if __name__ == "__main__":
    main()
