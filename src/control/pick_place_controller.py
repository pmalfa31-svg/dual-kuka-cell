"""
Controllore classico (macchina a stati + IK) per pick-and-place da nastro in movimento.

Nessun apprendimento: cinematica inversa ricalcolata ad ogni step sulla posizione
CORRENTE del pacco (closed-loop tracking). Gestisce nativamente un pacco che
arriva a una posizione laterale (Y) diversa ogni volta, perche' non fa altro
che "inseguire" lo stato attuale, non una posizione memorizzata.

Stati:
    IDLE       -> aspetta che il pacco entri nella finestra di presa
    TRACK      -> si porta in hover sopra il pacco, inseguendolo mentre si muove
    DESCEND    -> scende verso il pacco (continua a inseguire XY)
    GRASP      -> aggancia la ventosa
    LIFT       -> risale in quota con il pacco agganciato
    TRANSPORT  -> si porta in hover sopra la zona pallet
    PLACE      -> scende e rilascia
    RETURN     -> torna alla posa home, pronto per il prossimo pacco
    DONE       -> pacco depositato con successo
    MISSED     -> il pacco ha superato la finestra di presa senza essere agganciato
"""
import numpy as np
import mujoco

from src.control.ik_solver import solve_ik

JOINT_LOW = np.array([-3.05, -1.6, -2.6, -2.9, -2.9, -6.28])
JOINT_HIGH = np.array([3.05, 1.6, 2.6, 2.9, 2.9, 6.28])
HOME_Q = np.array([0.0, -1.33, 0.562, 1.341, 0.0, 0.0])
# Posa di "pronti" sopra il centro della finestra di presa (X~0.50, Y~0),
# altezza allineata ESATTAMENTE a HOVER_HEIGHT (0.40m sopra il nastro) usata
# sotto in TRACK/LIFT/TRANSPORT - un disallineamento anche di pochi cm qui
# impedisce a TRACK di soddisfare la tolleranza verticale per alcune posizioni
# di arrivo, causando pacchi mancati. A differenza di una posa verticale/
# ritratta lontana, questa riduce il tragitto per intercettare un pacco e le
# iterazioni IK necessarie ad ogni step (~0.15ms invece di ~8ms), evitando
# rallentamenti percepiti nel playback in tempo reale.

HOVER_HEIGHT = 0.40         # quota di sorvolo sopra il pacco/target (m) - alta abbastanza
                              # da garantire clearance a tutto il braccio (non solo all'EE)
                              # durante rotazioni ampie della base, sopra nastro/pallet (top=0.40m)
GRASP_HEIGHT_OFFSET = 0.015  # offset verticale del punto di presa sopra il centro pacco
XY_TOL_TRACK = 0.035         # tolleranza XY per considerare l'hover "raggiunto" (bersaglio in movimento)
Z_TOL_TRACK = 0.055          # margine per piccola deviazione residua di regolarizzazione
Z_TOL_LOADED = 0.10           # tolleranza verticale quando si porta il pacco: un servo PD
                              # senza termine integrale ha un errore di regime fisiologico
                              # sotto carico extra (drooping), va tollerato, non inseguito
SUCTION_THRESHOLD = 0.035   # soglia di aggancio ventosa (m) - stretta per evitare
MAX_JOINT_STEP = 0.08       # rad/step - moto fluido; con i guadagni PD alzati,
PICK_WINDOW_Y = (-0.15, 0.15)  # finestra di presa lungo il nastro (ora asse Y)
MISS_Y = 0.65                # oltre questa |Y| il pacco e' considerato mancato


class PickPlaceController:
    def __init__(self, model: mujoco.MjModel, target_pos: np.ndarray):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.target_pos = target_pos.copy()
        self.state = "IDLE"
        self.q_cmd = HOME_Q.copy()
        self._grasp_pos = None

    def reset(self):
        self.state = "IDLE"
        self.q_cmd = HOME_Q.copy()
        self._grasp_pos = None

    def _ik_step_toward(self, target_xyz, q_current):
        # NOTA: l'IK interno riparte sempre da HOME_Q, non da q_current.
        # Un solutore iterativo (DLS) per un braccio ridondante puo' rimanere
        # intrappolato in un minimo locale diverso a seconda del punto di
        # partenza; ripartire da una posa di riferimento nota (che nei test
        # converge sempre correttamente) evita l'intrappolamento. La velocita'
        # di movimento reale resta comunque governata dal clip sottostante,
        # quindi non si perde il controllo del moto.
        q_sol, converged = solve_ik(
            self.model, self.scratch, "ee_site", target_xyz, HOME_Q,
            JOINT_LOW, JOINT_HIGH, q_bias=HOME_Q, bias_weight=0.1,
        )
        delta = np.clip(q_sol - q_current, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        return q_current + delta, converged

    def step(self, pkg_pos: np.ndarray, ee_pos: np.ndarray):
        """Esegue un tick della macchina a stati. Ritorna (q_cmd, suction_on, done_flag)."""
        suction_on = self.state in ("GRASP", "LIFT", "TRANSPORT")

        if self.state == "IDLE":
            if PICK_WINDOW_Y[0] <= pkg_pos[1] <= PICK_WINDOW_Y[1]:
                self.state = "TRACK"
            elif pkg_pos[1] > MISS_Y:
                self.state = "MISSED"

        elif self.state == "TRACK":
            hover_target = pkg_pos + np.array([0, 0, HOVER_HEIGHT])
            self.q_cmd, _ = self._ik_step_toward(hover_target, self.q_cmd)
            xy_err = np.linalg.norm(ee_pos[:2] - pkg_pos[:2])
            if xy_err < XY_TOL_TRACK and abs(ee_pos[2] - hover_target[2]) < Z_TOL_TRACK:
                self.state = "DESCEND"
            if pkg_pos[1] > MISS_Y:
                self.state = "MISSED"

        elif self.state == "DESCEND":
            grasp_target = pkg_pos + np.array([0, 0, GRASP_HEIGHT_OFFSET])
            self.q_cmd, _ = self._ik_step_toward(grasp_target, self.q_cmd)
            dist = np.linalg.norm(ee_pos - pkg_pos)
            if dist < SUCTION_THRESHOLD:
                self.state = "GRASP"
            if pkg_pos[1] > MISS_Y:
                self.state = "MISSED"

        elif self.state == "GRASP":
            # Congelo la posizione di presa: da qui in poi il pacco e' solidale
            # con l'end-effector, quindi "pkg_pos" diventerebbe un bersaglio che
            # insegue se stesso se ricalcolato ogni step (inseguimento infinito).
            self._grasp_pos = pkg_pos.copy()
            self.state = "LIFT"

        elif self.state == "LIFT":
            lift_target = self._grasp_pos + np.array([0, 0, HOVER_HEIGHT])
            self.q_cmd, _ = self._ik_step_toward(lift_target, self.q_cmd)
            if ee_pos[2] > lift_target[2] - Z_TOL_LOADED:
                self.state = "TRANSPORT"

        elif self.state == "TRANSPORT":
            # Compenso l'offset EE->pacco: dopo una rotazione ampia del polso,
            # l'offset di presa (fisso nel frame locale) ruota con esso, quindi
            # ee_site e pacco non restano piu' co-locati. Miro con l'IK al punto
            # che porta il PACCO (non l'EE) sopra il target.
            ee_to_pkg_offset = ee_pos - pkg_pos
            hover_target = self.target_pos + np.array([0, 0, HOVER_HEIGHT]) + ee_to_pkg_offset
            self.q_cmd, _ = self._ik_step_toward(hover_target, self.q_cmd)
            xy_err = np.linalg.norm(pkg_pos[:2] - self.target_pos[:2])
            if xy_err < XY_TOL_TRACK and abs(pkg_pos[2] - self.target_pos[2] - HOVER_HEIGHT) < Z_TOL_LOADED:
                self.state = "PLACE"

        elif self.state == "PLACE":
            ee_to_pkg_offset = ee_pos - pkg_pos
            place_target = self.target_pos + np.array([0, 0, GRASP_HEIGHT_OFFSET]) + ee_to_pkg_offset
            self.q_cmd, _ = self._ik_step_toward(place_target, self.q_cmd)
            # Controllo la posizione del PACCO (non dell'EE): un offset fisso di
            # presa fa si' che l'EE resti sempre piu' in alto del centro pacco,
            # facendo sembrare "non ancora appoggiato" un pacco gia' a contatto
            # col pallet.
            dist_xy = np.linalg.norm(pkg_pos[:2] - self.target_pos[:2])
            if pkg_pos[2] < self.target_pos[2] + 0.05 and dist_xy < XY_TOL_TRACK:
                self.state = "RETURN"  # il chiamante disattiva la ventosa entrando qui

        elif self.state == "RETURN":
            # Non serve IK: conosciamo gia' gli angoli home, ci si muove
            # direttamente in spazio articolare (l'IK potrebbe convergere a
            # una soluzione ridondante diversa, cartesiano-equivalente ma
            # articolarmente lontana da HOME_Q, non facendo mai terminare lo stato).
            delta = np.clip(HOME_Q - self.q_cmd, -MAX_JOINT_STEP, MAX_JOINT_STEP)
            self.q_cmd = self.q_cmd + delta
            if np.linalg.norm(self.q_cmd - HOME_Q) < 0.05:
                self.state = "DONE"

        done = self.state in ("DONE", "MISSED")
        # La ventosa deve essere ON anche nell'ultimo step di DESCEND->GRASP transition
        suction_on = self.state in ("GRASP", "LIFT", "TRANSPORT", "PLACE")
        return self.q_cmd, suction_on, done
