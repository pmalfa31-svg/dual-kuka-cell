"""
Dimostrazione di pick-and-place classico (IK + macchina a stati) da nastro
in movimento, SENZA reinforcement learning. Il pacco arriva a una posizione
laterale (X) diversa ad ogni prova, e il controllore lo insegue in tempo reale.

Uso:
    python scripts/demo_ik_conveyor.py --trials 30 --render
    python scripts/demo_ik_conveyor.py --trials 100          # solo statistiche, no grafica
"""
import os
import sys
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mujoco

from src.control.pick_place_controller import PickPlaceController, HOME_Q

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "mujoco", "single_arm_cell.xml"
)

BELT_SPEED = 0.12          # m/s, velocita' del nastro
BELT_X_RANGE = (0.40, 0.60)   # variabilita' laterale di arrivo del pacco (ora sull'asse X)
PACKAGE_START_Y = -0.60
PACKAGE_Z = 0.44
MAX_SIM_SECONDS = 20.0


def compute_relpose(model, data, body1_name, body2_name):
    """Calcola la posa relativa REALE tra due corpi in questo istante.
    Necessario perche' il vincolo weld di MuJoCo, se non aggiornato, usa
    l'offset calcolato al momento della compilazione del modello (qpos0),
    non quello attuale - causando una correzione violenta se il pacco
    viene agganciato in una posizione diversa da quella di riferimento."""
    b1_pos, b1_quat = data.body(body1_name).xpos.copy(), data.body(body1_name).xquat.copy()
    b2_pos, b2_quat = data.body(body2_name).xpos.copy(), data.body(body2_name).xquat.copy()
    R1 = np.zeros(9)
    mujoco.mju_quat2Mat(R1, b1_quat)
    R1 = R1.reshape(3, 3)
    rel_pos_local = R1.T @ (b2_pos - b1_pos)
    quat1_inv = np.zeros(4)
    mujoco.mju_negQuat(quat1_inv, b1_quat)
    relquat = np.zeros(4)
    mujoco.mju_mulQuat(relquat, quat1_inv, b2_quat)
    return rel_pos_local, relquat


def run_trial(model, data, target_pos, seed, render_viewer=None):
    mujoco.mj_resetData(model, data)

    pkg_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "package_free")
    pkg_qpos_adr = model.jnt_qposadr[pkg_joint_id]
    pkg_dof_adr = model.jnt_dofadr[pkg_joint_id]
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_weld")

    rng = np.random.default_rng(seed)
    px = rng.uniform(*BELT_X_RANGE)

    data.qpos[:6] = HOME_Q
    data.ctrl[:6] = HOME_Q
    data.qpos[pkg_qpos_adr:pkg_qpos_adr + 3] = [px, PACKAGE_START_Y, PACKAGE_Z]
    data.qpos[pkg_qpos_adr + 3:pkg_qpos_adr + 7] = [1, 0, 0, 0]
    data.eq_active[eq_id] = 0
    mujoco.mj_forward(model, data)

    controller = PickPlaceController(model, target_pos)
    attached = False
    n_steps = int(MAX_SIM_SECONDS / model.opt.timestep)

    for step_i in range(n_steps):
        step_start_time = time.perf_counter()

        pkg_pos = data.body("package").xpos.copy()
        ee_pos = data.site("ee_site").xpos.copy()

        q_cmd, suction_on, done = controller.step(pkg_pos, ee_pos)
        data.ctrl[:6] = q_cmd

        dist = np.linalg.norm(ee_pos - pkg_pos)
        if suction_on and not attached and dist < 0.035:
            # Ricalcolo la posa relativa REALE prima di attivare il vincolo:
            # senza questo, MuJoCo userebbe l'offset di riferimento calcolato
            # al compile-time (nel nostro caso -0.73m!), causando una violenta
            # correzione istantanea che scaraventa via il pacco.
            rel_pos, relquat = compute_relpose(model, data, "link6", "package")
            eq_id_local = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "suction_weld")
            model.eq_data[eq_id_local, 3:6] = rel_pos
            model.eq_data[eq_id_local, 6:10] = relquat
            data.qvel[pkg_dof_adr:pkg_dof_adr + 6] = 0.0
            data.eq_active[eq_id] = 1
            attached = True
        elif not suction_on and attached:
            data.eq_active[eq_id] = 0
            attached = False

        # Nastro cinematico: mentre il pacco non e' agganciato, lo si fa scorrere
        # a velocita' costante lungo Y (perpendicolare al braccio), altezza fissa
        # (approssimazione semplice di un nastro trasportatore reale).
        if not attached:
            data.qvel[pkg_dof_adr:pkg_dof_adr + 6] = [0, BELT_SPEED, 0, 0, 0, 0]
            data.qpos[pkg_qpos_adr + 2] = PACKAGE_Z

        mujoco.mj_step(model, data)

        if render_viewer is not None:
            render_viewer.sync()
            # Pausa ADATTIVA: sottraggo il tempo gia' speso a calcolare (IK,
            # fisica, ecc.) dal budget del timestep, invece di dormire un tempo
            # fisso sopra un calcolo che puo' costare piu' del timestep stesso
            # (l'IK durante TRACK/DESCEND costa ~8ms, piu' dei 5ms di timestep:
            # una sleep fissa qui raddoppiava il tempo reale per step, facendo
            # sembrare il nastro "rallentare" proprio vicino alla zona di presa).
            elapsed = time.perf_counter() - step_start_time
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

        if done:
            final_dist_to_target = np.linalg.norm(
                data.body("package").xpos[:2] - target_pos[:2]
            )
            success = (controller.state == "DONE") and final_dist_to_target < 0.08
            return success, controller.state, step_i, px

    return False, "TIMEOUT", n_steps, px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(_MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # necessario: senza, le posizioni dei site sono zero
    target_pos = data.site("target_site").xpos.copy()

    viewer = None
    if args.render:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)

    results = []
    for trial in range(args.trials):
        success, final_state, n_steps, px = run_trial(model, data, target_pos, seed=trial, render_viewer=viewer)
        results.append(success)
        print(f"Prova {trial+1:3d}/{args.trials} | X arrivo={px:+.3f} | esito={'OK' if success else final_state:8s} | step={n_steps}")

    n_success = sum(results)
    print(f"\n[+] Tasso di successo: {n_success}/{args.trials} ({100*n_success/args.trials:.1f}%)")

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
