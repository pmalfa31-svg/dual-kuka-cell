import time
import numpy as np
import mujoco
import mujoco.viewer

from src.core.conveyor import OvalConveyor
from src.controllers.vacuum import VacuumController

def main():
    model = mujoco.MjModel.from_xml_path("assets/scene.xml")
    data = mujoco.MjData(model)

    conveyor = OvalConveyor(speed=0.6)
    vacuum = VacuumController(model, data)

    # Pose di default (Homeworld)
    q_r1_home = [0.0, -1.2, 1.8, 0.0, -0.6, 0.0]
    q_r2_home = [0.0, -1.2, 1.8, 0.0, -0.6, 0.0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_start = time.time()
        while viewer.is_running():
            step_start = time.time()

            # 1. Aggiornamento dinamico del nastro
            conveyor.apply_conveyor_velocity(model, data, "box_0")

            # 2. Controllo posizione dei robot
            t = time.time() - t_start
            # Oscillazione di test sui primi giunti
            data.ctrl[0:6] = q_r1_home + np.array([0.3 * np.sin(t), 0.1 * np.cos(t), 0, 0, 0, 0])
            data.ctrl[6:12] = q_r2_home + np.array([0.3 * np.sin(t + np.pi), 0.1 * np.cos(t), 0, 0, 0, 0])

            # 3. Avanzamento fisico
            mujoco.mj_step(model, data)
            viewer.sync()

            # Mantenimento del framerate real-time (500 Hz physics / 60 FPS viewer)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
