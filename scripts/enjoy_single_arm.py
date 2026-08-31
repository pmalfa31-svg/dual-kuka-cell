"""
Visualizza in 3D il comportamento della policy addestrata (Fase 1, singolo braccio).

Uso:
    python scripts/enjoy_single_arm.py --model checkpoints/single_arm/best/best_model.zip
"""
import os
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.envs.single_arm_env import SingleArmPalletizerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/single_arm/best/best_model.zip")
    parser.add_argument("--vecnormalize", type=str,
                         default="checkpoints/single_arm/best/vecnormalize.pkl",
                         help="Statistiche di normalizzazione salvate durante il training. "
                              "Obbligatorio: senza, la policy vede osservazioni fuori scala.")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    base_env = DummyVecEnv([lambda: SingleArmPalletizerEnv(render_mode="human")])
    if os.path.exists(args.vecnormalize):
        env = VecNormalize.load(args.vecnormalize, base_env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[!] ATTENZIONE: {args.vecnormalize} non trovato. Procedo senza normalizzazione "
              f"(la policy potrebbe comportarsi male se e' stata addestrata con VecNormalize).")
        env = base_env

    model = PPO.load(args.model, device="cpu")

    successes = 0
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        done = [False]
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            ep_reward += reward[0]
            time.sleep(0.01)
        info = infos[0]
        successes += int(info.get("success", False))
        print(f"Episodio {ep}/{args.episodes} | reward: {ep_reward:.2f} | successo: {info.get('success', False)}")

    print(f"\n[+] Tasso di successo: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
