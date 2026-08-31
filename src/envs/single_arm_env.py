"""
Fase 1 - Ambiente a singolo braccio (pick-and-place statico).

Un solo KUKA a 6 DOF deve raggiungere un pacco fermo sul tavolo,
agganciarlo con la ventosa (suction) e trasportarlo sopra la zona
pallet target. Questo ambiente serve a validare reward, osservazioni
e dinamica di aggancio PRIMA di introdurre nastro e secondo braccio.

Azione (7D, continua in [-1, 1]):
    a[0:6] -> delta di posizione articolare normalizzato sui 6 giunti
              (applicato come += a*MAX_JOINT_DELTA a un target inseguito da servo PD)
    a[6]   -> comando ventosa: >0.5 tenta aggancio, <=0.5 rilascia

Osservazione:
    qpos giunti (6) + qvel giunti (6) + pos end-effector (3)
    + pos pacco relativa a EE (3) + pos pacco assoluta (3)
    + pos target relativa a pacco (3) + stato ventosa (1) + distanza EE-target (1)
    = 26 dimensioni
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "mujoco", "single_arm_cell.xml"
)

# Soglia di aggancio ventosa (metri) - vedi README originale: 8 cm
SUCTION_THRESHOLD = 0.08

# Massimo delta di posizione articolare per step (rad) - come da spec originale
MAX_JOINT_DELTA = 0.05

# Limiti articolari (devono combaciare con gli attributi "range" del MJCF)
JOINT_LOW = np.array([-3.05, -1.6, -2.6, -2.9, -2.9, -6.28], dtype=np.float32)
JOINT_HIGH = np.array([3.05, 1.6, 2.6, 2.9, 2.9, 6.28], dtype=np.float32)


class SingleArmPalletizerEnv(gym.Env):
    metadata = {"render_modes": ["human", None], "render_fps": 50}

    def __init__(self, render_mode: str | None = None, max_steps: int = 300, difficulty: float = 1.0):
        """
        difficulty: 0.0 -> pacco spawna vicino alla posa home (task facile, per bootstrap)
                    1.0 -> pacco spawna nel range completo originale (task finale)
        """
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._step_count = 0
        self._viewer = None

        self._eq_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_weld"
        )
        self._package_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "package"
        )
        self._package_qpos_adr = self.model.body_jntadr[self._package_body_id]
        self._package_qpos_start = self.model.jnt_qposadr[self._package_qpos_adr]

        self._target_pos = self.data.site("target_site").xpos.copy()
        self._attached = False
        self._prev_ee_to_pkg = None
        self._q_target = None  # target di posizione articolare inseguito dai servo PD

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        # Range di reset "difficile" (task finale, come da spec originale)
        self._package_range_hard = {"x": (0.45, 0.65), "y": (0.15, 0.40), "z": 0.44}
        # Range di reset "facile" (vicino alla posa home, EE~[0.81, 0.0, 0.43])
        # usato per il curriculum: bootstrap del comportamento di reach.
        self._package_range_easy = {"x": (0.72, 0.85), "y": (-0.05, 0.15), "z": 0.44}

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        d = self.data
        joint_qpos = d.qpos[:6].astype(np.float32)
        joint_qvel = d.qvel[:6].astype(np.float32)
        ee_pos = d.site("ee_site").xpos.astype(np.float32)
        pkg_pos = d.body("package").xpos.astype(np.float32)
        ee_to_pkg = (pkg_pos - ee_pos).astype(np.float32)
        pkg_to_target = (self._target_pos - pkg_pos).astype(np.float32)
        ee_to_target = np.array([np.linalg.norm(ee_pos - self._target_pos)], dtype=np.float32)
        suction_state = np.array([1.0 if self._attached else 0.0], dtype=np.float32)

        obs = np.concatenate([
            joint_qpos, joint_qvel, ee_pos, ee_to_pkg, pkg_pos,
            pkg_to_target, suction_state, ee_to_target,
        ])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Posiziona braccio a una posa "home" ragionevole con piccolo rumore
        home_qpos = np.array([0.0, -0.3, 0.6, 0.0, 0.3, 0.0])
        noise = self.np_random.uniform(-0.05, 0.05, size=6)
        self.data.qpos[:6] = home_qpos + noise
        self._q_target = (home_qpos + noise).astype(np.float32)
        self.data.ctrl[:6] = self._q_target

        # Posiziona il pacco casualmente sul tavolo, interpolando facile/difficile
        e, h = self._package_range_easy, self._package_range_hard
        x_lo = e["x"][0] + self.difficulty * (h["x"][0] - e["x"][0])
        x_hi = e["x"][1] + self.difficulty * (h["x"][1] - e["x"][1])
        y_lo = e["y"][0] + self.difficulty * (h["y"][0] - e["y"][0])
        y_hi = e["y"][1] + self.difficulty * (h["y"][1] - e["y"][1])
        px = self.np_random.uniform(x_lo, x_hi)
        py = self.np_random.uniform(y_lo, y_hi)
        pz = h["z"]
        qs = self._package_qpos_start
        self.data.qpos[qs:qs + 3] = [px, py, pz]
        self.data.qpos[qs + 3:qs + 7] = [1, 0, 0, 0]  # quaternione identita'

        self.model.eq_active0[self._eq_id] = 0
        self.data.eq_active[self._eq_id] = 0
        self._attached = False

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._prev_ee_to_pkg = np.linalg.norm(
            self.data.site("ee_site").xpos - self.data.body("package").xpos
        )
        # Diagnostica episodio: quanto si e' avvicinato il braccio, se ha mai agganciato
        self._episode_min_dist = float(self._prev_ee_to_pkg)
        self._episode_ever_latched = False

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        joint_delta, suction_cmd = action[:6], action[6]

        self._q_target = np.clip(
            self._q_target + joint_delta * MAX_JOINT_DELTA, JOINT_LOW, JOINT_HIGH
        )
        self.data.ctrl[:6] = self._q_target
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Rete di sicurezza: se nonostante il controllo PD la fisica diverge
        # (contatti rigidi, configurazioni estreme), termino l'episodio con
        # penalita' invece di propagare stato/osservazioni corrotte al buffer PPO.
        if not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            obs = np.nan_to_num(self._get_obs(), nan=0.0, posinf=0.0, neginf=0.0)
            info = {"success": False, "unstable": True}
            return obs, -20.0, True, False, info

        ee_pos = self.data.site("ee_site").xpos
        pkg_pos = self.data.body("package").xpos
        dist_ee_pkg = float(np.linalg.norm(ee_pos - pkg_pos))

        just_latched = False
        just_released = False

        if not self._attached and suction_cmd > 0.5 and dist_ee_pkg < SUCTION_THRESHOLD:
            self.data.eq_active[self._eq_id] = 1
            self._attached = True
            just_latched = True
            self._episode_ever_latched = True
            self._grab_pos = pkg_pos.copy()
        elif self._attached and suction_cmd <= 0.5:
            self.data.eq_active[self._eq_id] = 0
            self._attached = False
            just_released = True

        self._episode_min_dist = min(self._episode_min_dist, dist_ee_pkg)

        dist_pkg_target = float(np.linalg.norm(pkg_pos - self._target_pos))

        reward, terminated, info = self._compute_reward(
            dist_ee_pkg, dist_pkg_target, just_latched, just_released, action
        )
        self._prev_ee_to_pkg = dist_ee_pkg

        truncated = self._step_count >= self.max_steps
        obs = self._get_obs()

        if terminated or truncated:
            info["ep_diag"] = {
                "min_dist_ee_pkg": self._episode_min_dist,
                "ever_latched": self._episode_ever_latched,
            }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _compute_reward(self, dist_ee_pkg, dist_pkg_target, just_latched, just_released, action):
        info = {"success": False, "latched": self._attached}
        terminated = False

        # 1. Reach: shaping potenziale su avvicinamento EE->pacco.
        #    IMPORTANTE: va applicato SEMPRE (non solo se non agganciato), altrimenti
        #    lo shaping potenziale (Ng et al.) perde la proprieta' di telescopare
        #    correttamente: ogni sgancio "resetta" la distanza precedente vicino a
        #    zero, e un allontanamento successivo genera un salto fortemente negativo
        #    ripetuto ad ogni ciclo aggancio/sgancio. Quando agganciato, dist_ee_pkg
        #    resta comunque ~0 (vincolo weld), quindi il termine e' naturalmente ~0.
        r_reach = 8.0 * (self._prev_ee_to_pkg - dist_ee_pkg)

        # 2. Latch: bonus una tantum all'aggancio riuscito
        r_latch = 15.0 if just_latched else 0.0

        # 3. Transport: shaping su avvicinamento pacco->target (solo se agganciato)
        r_transport = 0.0
        if self._attached:
            r_transport = 3.0 * max(0.0, (self._prev_pkg_target_dist_or(dist_pkg_target) - dist_pkg_target))
        self._last_pkg_target_dist = dist_pkg_target

        # 4. Palletizzazione riuscita
        r_pallet = 0.0
        pkg_z = self.data.body("package").xpos[2]
        table_z = 0.40
        on_target_xy = np.linalg.norm(
            self.data.body("package").xpos[:2] - self._target_pos[:2]
        ) < 0.06
        settled_on_table = abs(pkg_z - table_z) < 0.06
        if self._attached and on_target_xy and settled_on_table:
            r_pallet = 100.0
            info["success"] = True
            terminated = True

        # 5. Penalita' rilascio prematuro SOLO se il pacco era davvero sollevato in aria
        #    e lontano dal target (evita "drop and abandon" vero). Prima penalizzava
        #    anche il semplice sfarfallio casuale del comando ventosa su un pacco
        #    ancora fermo sul tavolo, dominando il reward con rumore non informativo.
        pkg_z_now = pkg_z
        was_airborne = pkg_z_now > table_z + 0.03
        r_drop_penalty = 3.0 if (just_released and was_airborne and dist_pkg_target > 0.10) else 0.0

        # 6. Regolarizzazione motoria (penalita' su comandi ampi -> movimenti piu' fluidi)
        #    Peso ridotto (0.01 -> 0.002): prima rischiava di dominare un reach reward
        #    gia' debole, scoraggiando l'esplorazione necessaria a trovare il pacco.
        r_torque = 0.002 * float(np.sum(np.square(action[:6])))

        # 7. Penalita' collisione pacco/EE con il pavimento (fuori dal tavolo -> episodio fallito)
        r_collision = 0.0
        if pkg_z < 0.15:
            r_collision = 20.0
            terminated = True
            info["success"] = False

        reward = (
            r_reach + r_latch + r_transport + r_pallet
            - r_drop_penalty - r_torque - r_collision
        )
        info.update({
            "r_reach": r_reach, "r_latch": r_latch, "r_transport": r_transport,
            "r_pallet": r_pallet, "r_torque": r_torque, "r_collision": r_collision,
        })
        return reward, terminated, info

    def _prev_pkg_target_dist_or(self, fallback):
        return getattr(self, "_last_pkg_target_dist", fallback)

    # ------------------------------------------------------------------
    def render(self):
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
