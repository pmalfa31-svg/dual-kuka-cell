import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import robosuite as suite
from tqdm import tqdm
from src.models.policy import BimanualPolicy

def run_benchmark(num_episodes: int = 20, horizon: int = 150):
    print(f"[*] Inizializzazione benchmark su {num_episodes} episodi...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=False,          # Headless per massima velocita
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=horizon,
        use_camera_obs=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BimanualPolicy(obs_dim=121, action_dim=14).to(device)
    model.load_state_dict(torch.load("data/bimanual_policy.pt", map_location=device))
    model.eval()

    successes = 0
    completion_steps = []

    pbar = tqdm(range(1, num_episodes + 1), desc="[Benchmark]", unit="ep")

    for ep in pbar:
        obs = env.reset()
        ep_success = False
        
        for step in range(horizon):
            flat_obs = np.concatenate([
                obs["robot0_proprio-state"],
                obs["robot1_proprio-state"],
                obs["object-state"]
            ], dtype=np.float32)

            with torch.no_grad():
                tensor_obs = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(tensor_obs).squeeze(0).cpu().numpy()

            obs, reward, done, _ = env.step(action)

            # In TwoArmLift il task e completato con successo quando l'oggetto e sollevato stabilmente
            if done or reward > 0.9:
                ep_success = True
                completion_steps.append(step)
                break

        if ep_success:
            successes += 1
            
        pbar.set_postfix({"Success Rate": f"{(successes / ep) * 100:.1f}%"})

    env.close()

    success_rate = (successes / num_episodes) * 100
    avg_steps = np.mean(completion_steps) if completion_steps else 0.0

    print("\n" + "="*50)
    print(" RISULTATI BENCHMARK POLICY (Behavioral Cloning MLP)")
    print("="*50)
    print(f"Episodi Totali        : {num_episodes}")
    print(f"Successi              : {successes}/{num_episodes}")
    print(f"Success Rate          : {success_rate:.2f}%")
    print(f"Step Medi Completamento: {avg_steps:.1f} frames")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
