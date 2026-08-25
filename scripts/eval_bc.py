import torch
import torch.nn as nn
import numpy as np
import robosuite as suite
import time

class BimanualPolicy(nn.Module):
    """Architettura della policy neurale bimanuale."""
    def __init__(self, obs_dim: int, action_dim: int = 14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def main():
    print("[*] Inizializzazione ambiente e caricamento policy neurale...")
    
    env = suite.make(
        env_name="TwoArmLift",
        robots=["IIWA", "IIWA"],
        has_renderer=True,          # Finestra 3D interattiva
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=150,
        use_camera_obs=False
    )

    obs_dim = 121
    action_dim = 14
    
    device = torch.device("cpu")
    model = BimanualPolicy(obs_dim, action_dim).to(device)
    model.load_state_dict(torch.load("data/bimanual_policy.pt", map_location=device))
    model.eval()

    num_eval_episodes = 3
    print(f"\n[+] Policy caricata. Avvio di {num_eval_episodes} episodi di test autonomi...")

    for ep in range(1, num_eval_episodes + 1):
        obs = env.reset()
        print(f"\n--- Episodio {ep}/{num_eval_episodes} ---")
        
        ep_reward = 0.0
        for step in range(150):
            # Costruzione vettore di osservazione piatto
            flat_obs = np.concatenate([
                obs["robot0_proprio-state"],
                obs["robot1_proprio-state"],
                obs["object-state"]
            ], dtype=np.float32)

            # Inferenza della policy neurale
            with torch.no_grad():
                obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
                action = model(obs_tensor).squeeze(0).cpu().numpy()

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(0.01)  # Limita il framerate per visualizzazione fluida

            if done:
                print(f"[+] Task completato con successo al frame {step}!")
                break

        print(f"Episodio {ep} terminato | Reward cumulativo: {ep_reward:.2f}")

    env.close()
    print("\n[+] Valutazione completata.")

if __name__ == "__main__":
    main()
