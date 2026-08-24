import os
import numpy as np
from tqdm import tqdm
from src.envs.dual_arm_env import DualKukaPalletizeEnv
from src.core.expert import IndustrialExpertPlanner

def main():
    env = DualKukaPalletizeEnv()
    expert = IndustrialExpertPlanner(env.pallet_1, env.pallet_2)
    
    observations = []
    actions = []
    target_episodes = 50

    pbar = tqdm(total=target_episodes, desc="[Record] Generazione Dataset Esperti", unit="ep")

    successful_episodes = 0
    while successful_episodes < target_episodes:
        obs, _ = env.reset()
        ep_obs, ep_act = [], []
        done = False

        while not done:
            action = expert.get_expert_action(env)
            ep_obs.append(obs)
            ep_act.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if info["is_success"]:
                successful_episodes += 1
                observations.extend(ep_obs)
                actions.extend(ep_act)
                pbar.update(1)
                pbar.set_postfix({"Transizioni": len(observations)})
                break

    pbar.close()

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(
        "data/expert_dataset.npz",
        obs=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32)
    )
    print(f"\n[+] Dataset generato con successo: {len(observations)} transizioni ottime.")

if __name__ == "__main__":
    main()
