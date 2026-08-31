"""
Cinematica inversa (IK) tramite minimi quadrati smorzati (Damped Least Squares).

Approccio standard in robotica: dato un target di posizione 3D per l'end-effector,
calcola iterativamente gli angoli articolari che lo raggiungono, usando lo Jacobiano
di posizione (derivata di pos_EE rispetto a qpos) fornito nativamente da MuJoCo.

Nota: risolviamo solo la POSIZIONE (3 vincoli, 6 incognite = problema ridondante).
Per un end-effector a ventosa (non un gripper a due dita) l'orientamento e' meno
critico: la ventosa fa presa per contatto puntuale, non serve un allineamento
angolare preciso. Usiamo un termine di regolarizzazione (bias verso una posa di
riferimento) per rendere la soluzione ripetibile invece che arbitraria tra le
infinite soluzioni possibili.
"""
import numpy as np
import mujoco


def solve_ik(
    model: mujoco.MjModel,
    scratch_data: mujoco.MjData,
    site_name: str,
    target_pos: np.ndarray,
    q_init: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    q_bias: np.ndarray | None = None,
    n_joints: int = 6,
    max_iters: int = 150,
    tol: float = 5e-4,
    damping: float = 0.05,
    step_scale: float = 1.0,
    bias_weight: float = 0.02,
):
    """
    Risolve IK per site_name verso target_pos partendo da q_init.

    scratch_data: un MjData "di scorta" separato da quello della simulazione
        reale, usato solo per iterare la cinematica senza toccare lo stato
        fisico vero (nessun mj_step, solo mj_forward per la cinematica).

    Ritorna: (q_solution, converged: bool)
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    q = q_init.copy().astype(np.float64)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    converged = False
    for _ in range(max_iters):
        scratch_data.qpos[:n_joints] = q
        mujoco.mj_forward(model, scratch_data)
        cur_pos = scratch_data.site(site_id).xpos.copy()
        err = target_pos - cur_pos
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            converged = True
            break

        mujoco.mj_jacSite(model, scratch_data, jacp, jacr, site_id)
        J = jacp[:, :n_joints]  # 3 x n_joints

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * err
        lam2 = damping ** 2
        JJt = J @ J.T + lam2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)

        # Regolarizzazione nello spazio nullo: spinge dolcemente verso q_bias
        # senza compromettere la soluzione di posizione (proiettore nello spazio nullo).
        if q_bias is not None:
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(3))
            N = np.eye(n_joints) - J_pinv @ J
            dq += bias_weight * (N @ (q_bias - q))

        q = q + step_scale * dq
        q = np.clip(q, joint_low, joint_high)

    return q, converged
