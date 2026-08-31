"""
Fase 1 - Training PPO su ambiente a singolo braccio (CPU-only).

Uso:
    python scripts/train_single_arm.py --timesteps 2000000 --n-envs 8

Consigli pratici:
    - --n-envs: metti circa (num_core_fisici - 1). Su 6 core reali, prova 5-6.
    - Lascia girare in background e controlla i progressi con:
        tensorboard --logdir logs/single_arm
    - I checkpoint si salvano periodicamente in checkpoints/single_arm/,
      cosi' se il PC si spegne o il processo crasha non perdi tutto.
"""
import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from collections import deque

from src.envs.single_arm_env import SingleArmPalletizerEnv


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """EvalCallback salva best_model.zip ma NON le statistiche di VecNormalize.
    Senza questo, il modello migliore verrebbe valutato/usato con normalizzazione
    sbagliata in un secondo momento (es. in enjoy_single_arm.py)."""

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir

    def _on_step(self) -> bool:
        os.makedirs(self.save_dir, exist_ok=True)
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(os.path.join(self.save_dir, "vecnormalize.pkl"))
        return True


class DiagnosticsCallback(BaseCallback):
    """Logga su tensorboard cosa succede DAVVERO dentro gli episodi:
    quanto si avvicina il braccio al pacco e quanto spesso aggancia.
    Senza questo, un ep_rew_mean piatto o basso non dice se il problema
    e' il reach, l'aggancio, o il trasporto."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.min_dist_buf = deque(maxlen=window)
        self.latch_buf = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            diag = info.get("ep_diag")
            if diag is not None:
                self.min_dist_buf.append(diag["min_dist_ee_pkg"])
                self.latch_buf.append(float(diag["ever_latched"]))
        if len(self.min_dist_buf) > 0 and self.n_calls % 500 == 0:
            self.logger.record("diagnostics/min_dist_ee_pkg", sum(self.min_dist_buf) / len(self.min_dist_buf))
            self.logger.record("diagnostics/latch_rate", sum(self.latch_buf) / len(self.latch_buf))
        return True


def make_env(difficulty: float = 1.0):
    def _init():
        env = SingleArmPalletizerEnv(difficulty=difficulty)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                         help="Numero totale di step di ambiente da raccogliere.")
    parser.add_argument("--n-envs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                         help="Numero di ambienti paralleli (CPU-bound: lascia 1 core libero per il SO).")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                         help="Ogni quanti step (per env) salvare un checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                         help="Percorso di un checkpoint .zip da cui riprendere il training.")
    parser.add_argument("--difficulty", type=float, default=1.0,
                         help="0.0 = pacco vicino alla posa home (bootstrap del reach), "
                              "1.0 = range completo originale. Usa 0.0 per il primo run, "
                              "poi 1.0 riprendendo dal checkpoint (--resume) per generalizzare.")
    args = parser.parse_args()

    os.makedirs("checkpoints/single_arm", exist_ok=True)
    os.makedirs("logs/single_arm", exist_ok=True)

    print(f"[*] Avvio training PPO Fase 1 (singolo braccio)")
    print(f"[*] Ambienti paralleli: {args.n_envs} | Step totali: {args.timesteps:,}")

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = make_vec_env(make_env(args.difficulty), n_envs=args.n_envs, vec_env_cls=vec_env_cls)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)

    eval_env = make_vec_env(make_env(args.difficulty), n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    if args.resume:
        print(f"[*] Ripresa training da checkpoint: {args.resume}")
        vecnorm_path = args.resume.replace(".zip", "") + "_vecnormalize.pkl"
        vecnorm_path_alt = os.path.join(os.path.dirname(args.resume), "vecnormalize_final.pkl")
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env.venv)
        elif os.path.exists(vecnorm_path_alt):
            env = VecNormalize.load(vecnorm_path_alt, env.venv)
        else:
            print("[!] ATTENZIONE: nessun file VecNormalize trovato accanto al checkpoint. "
                  "Le statistiche di normalizzazione ripartiranno da zero: possibile instabilita' iniziale.")
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            n_steps=1024,          # rollout per env prima di ogni update
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,        # incoraggia esplorazione, utile nella fase iniziale
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            tensorboard_log="logs/single_arm",
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path="checkpoints/single_arm",
        name_prefix="ppo_single_arm",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/single_arm/best",
        log_path="logs/single_arm/eval",
        eval_freq=max(25_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeOnBestCallback("checkpoints/single_arm/best"),
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, DiagnosticsCallback()],
        progress_bar=True,
    )

    model.save("checkpoints/single_arm/final_model")
    env.save("checkpoints/single_arm/vecnormalize_final.pkl")
    print("[+] Training completato. Modello finale in checkpoints/single_arm/final_model.zip")
    print("[+] Statistiche di normalizzazione in checkpoints/single_arm/vecnormalize_final.pkl")
    print("[+] Modello migliore (per reward di eval) in checkpoints/single_arm/best/best_model.zip")


if __name__ == "__main__":
    main()
