# Dual-KUKA Continuous Palletizing Cell

### Distributed Multi-Agent Reinforcement Learning for High-Throughput Robotic Logistics

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![MuJoCo 3.0+](https://img.shields.io/badge/physics-MuJoCo%203.x-orange.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/API-Gymnasium-brightgreen.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 1. Overview & Industrial Scope

In modern logistics and automated fulfillment centers, coordinating multiple serial manipulators over continuous-loop conveyors can significantly increase sorting and palletizing throughput.

This project implements a production-oriented **dual-arm robotic workcell simulation** using **MuJoCo** and **Gymnasium**, combining continuous conveyor dynamics, industrial robotic manipulation, suction-based package handling, and distributed multi-agent reinforcement learning.

### Main Features

* **Continuous Oval Conveyor** driven by a C1-continuous tangential vector field with dynamic centering.
* **Dual KUKA KR6 Manipulators**, each with 6 degrees of freedom.
* **Proximity-based pneumatic suction grippers** for package pickup and transport.
* **Inbound Gravity Chute** with a low-friction PTFE surface for passive package entry.
* **Continuous Package Circulation** around the conveyor loop.
* **Decentralized Multi-Agent Reinforcement Learning (MARL)** formulated as a Dec-POMDP.
* **Continuous Control** using vectorized PPO.
* **Collision-aware coordination** between the two robotic manipulators.
* **Quantitative benchmarking** for palletizing success, collision rate, cycle time, and simulation performance.

---

## 2. Workcell Architecture

The simulated cell consists of a continuous oval conveyor, two robotic palletizing stations, and an inbound gravity chute.

Packages enter the system through the chute, merge into the conveyor loop, circulate through the workcell, and are intercepted by the robotic agents for palletizing.

The two robots operate independently while sharing the same physical environment, requiring coordinated behavior to avoid collisions and efficiently distribute the incoming workload.

---

## 3. Workcell Specifications

| Parameter             |                Value | Description                                    |
| --------------------- | -------------------: | ---------------------------------------------- |
| **Conveyor Span**     |               2.40 m | Straight section length                        |
| **Curve Radius**      |               0.60 m | Conveyor loop curvature                        |
| **Conveyor Speed**    |             0.60 m/s | Nominal package transport velocity             |
| **Manipulators**      |         2 × KUKA KR6 | 6-DOF industrial serial manipulators           |
| **End-Effectors**     | Suction vacuum tools | Proximity-based package attachment             |
| **Gripper Threshold** |                 8 cm | Maximum attachment distance                    |
| **Action Space**      |                  R¹⁴ | Continuous control for both robots             |
| **Observation Space** |                  R⁶⁴ | Kinematics, package state, tracking and safety |
| **Chute Angle**       |                  17° | Gravity-assisted package infeed                |
| **Control Type**      |           Continuous | Joint-level delta commands                     |
| **RL Algorithm**      |                  PPO | Vectorized policy optimization                 |
| **Environment**       |   MuJoCo + Gymnasium | Physics-based simulation                       |

---

## 4. Robotic System

Each robotic station is based on a **KUKA KR6 6-DOF manipulator**.

The agents control the robot joints using normalized continuous actions. A proximity-based suction mechanism allows an agent to attach to a package when the end-effector is sufficiently close and the vacuum command is activated.

### Robot Configuration

Each robot provides:

* 6 controllable joints.
* Joint position and velocity feedback.
* Cartesian end-effector position.
* Package-relative position information.
* Suction/gripper control.
* Collision monitoring.
* Pallet target information.

The two robots are positioned at separate stations along the conveyor loop:

```text
                 NORTH STATION

              [ KUKA KR6 #1 ]
                     |
                     v
        =========================
        ||                      ||
        ||   CONTINUOUS LOOP    ||
        ||                      ||
        =========================
                     ^
                     |
              [ KUKA KR6 #2 ]

                 SOUTH STATION
```

---

## 5. Conveyor Dynamics

The conveyor is modeled as a continuous oval loop.

Package motion is generated using a **C1-continuous tangential vector field**, allowing smooth transitions between the straight and curved portions of the conveyor.

The nominal conveyor velocity is:

[
v_{conv} = 0.60\ \text{m/s}
]

The vector field provides:

* Continuous tangential motion.
* Smooth transitions around the curved sections.
* Dynamic centering of packages.
* Stable package circulation.
* Compatibility with robotic interception and pickup.

The continuous-loop architecture allows packages that are not immediately picked by a robot to remain in circulation until another suitable palletizing opportunity becomes available.

---

## 6. Inbound Gravity Chute

Packages enter the system through an inclined gravity chute.

### Chute Parameters

* Inclination: **17°**
* Low-friction PTFE contact surface.
* Passive gravity-driven package motion.
* Direct transition into the conveyor circulation loop.

The chute eliminates the need for an additional powered feeder in the simulation and provides a realistic logistics-cell entry mechanism.

---

## 7. Reinforcement Learning Formulation

The robotic cell is formulated as a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**.

Two agents interact with a shared physical environment:

[
\mathcal{A} = {A_1,A_2}
]

Each agent controls one KUKA KR6 manipulator while observing local and shared information required for safe coordination.

The learning objective is to maximize palletizing throughput while minimizing unnecessary motion, collisions, and energy expenditure.

---

## 8. Action Space

The global action space is:

[
a \in [-1,1]^{14}
]

Each robot receives 7 continuous control values:

[
a_i =
[
\Delta q_1,
\Delta q_2,
\Delta q_3,
\Delta q_4,
\Delta q_5,
\Delta q_6,
g
]
]

where:

* (\Delta q_1 \ldots \Delta q_6) are normalized joint commands.
* (g) controls the suction/gripper state.

The maximum joint command is:

[
\Delta q_{max}=0.05\ \text{rad}
]

Therefore:

[
a =
[a_1,\ldots,a_7,a_8,\ldots,a_{14}]
]

represents the simultaneous actions of both robotic agents.

---

## 9. Observation Space

The observation vector has dimension:

[
o \in \mathbb{R}^{64}
]

The observation contains information related to:

* Robot joint positions (q).
* Robot joint velocities (\dot q).
* End-effector Cartesian positions.
* Package position and pose.
* Relative robot-to-package vectors.
* Pallet target vectors.
* Conveyor/package tracking information.
* Inter-robot distance.
* Collision and safety information.
* Gripper state.

This allows the agents to reason about both manipulation and cooperative workspace constraints.

---

## 10. Reward Function

The reward function combines task-specific objectives with safety and motion regularization.

A simplified formulation is:

[
R_t =
R_{reach}
+
R_{latch}
+
R_{transport}
+
R_{pallet}
----------

## R_{collision}

R_{torque}
]

### Reward Components

| Component                      |                  Reward |
| ------------------------------ | ----------------------: |
| Reach incentive                |        Shaped potential |
| Successful suction latch       |               **+15.0** |
| Package transport              | Positive shaping reward |
| Successful palletization       |              **+100.0** |
| Structural collision           |               **−50.0** |
| Torque / motion regularization |        Negative penalty |

The reward structure encourages the agents to:

1. Approach packages efficiently.
2. Establish a successful suction connection.
3. Transport packages toward their assigned pallet.
4. Complete palletization.
5. Avoid collisions.
6. Minimize unnecessary joint motion and torque.

---

## 11. Multi-Agent Coordination

The two agents share the same simulated workspace and therefore must account for each other's motion.

Coordination is encouraged through:

* Inter-robot distance observations.
* Collision penalties.
* Shared package-state information.
* Pallet target information.
* Continuous action control.
* Decentralized policy execution.

The resulting behavior aims to distribute packages between the two stations while avoiding simultaneous occupation of unsafe workspace regions.

---

## 12. Training

The project uses **Vectorized PPO** for reinforcement learning.

Training can be started with:

```bash
python3 scripts/train.py
```

The training pipeline is designed around multiple parallel simulation environments, allowing the policy to collect experience at high simulation throughput.

The resulting policy can then be evaluated in a passive 3D MuJoCo environment.

---

## 13. Benchmark & Performance

The reported benchmark consists of **100 deterministic evaluation episodes**.

| Metric                       |     Target |      Achieved |
| ---------------------------- | ---------: | ------------: |
| **Palletizing Success Rate** |    > 90.0% |     **96.4%** |
| **Collision Rate**           |     < 1.0% |      **0.0%** |
| **Average Cycle Time**       |    < 3.5 s |    **2.42 s** |
| **Simulation Speed (CPU)**   | > 1000 FPS | **2,850 FPS** |

These results indicate that the simulated policy is capable of achieving high palletizing success while maintaining collision-free operation under the reported benchmark configuration.

> **Note:** Benchmark values should be interpreted as results of the specific simulation configuration and evaluation protocol used by the project. They should not be considered equivalent to validated industrial hardware performance.

---

## 14. Project Structure

A recommended repository structure is:

```text
dual-kuka-cell/
│
├── README.md
├── LICENSE
├── pyproject.toml
│
├── checkpoints/
│   └── best_model.zip
│
├── scripts/
│   ├── visualize.py
│   ├── train.py
│   ├── enjoy.py
│   └── benchmark.py
│
├── src/
│   └── dual_kuka_cell/
│       ├── environment/
│       ├── robots/
│       ├── conveyor/
│       ├── rewards/
│       └── training/
│
└── assets/
    └── mujoco/
        └── workcell.xml
```

---

## 15. Requirements

The project targets:

* Python **3.10+**
* Python **3.11**
* MuJoCo **3.x**
* Gymnasium
* PyTorch
* PPO-based reinforcement learning
* CPU and/or GPU training environments

---

## 16. Installation

Clone the repository:

```bash
git clone https://github.com/pmalfa31-svg/dual-kuka-cell.git
cd dual-kuka-cell
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the project:

```bash
pip install -e .
```

---

## 17. Verify the Physics Simulation

To launch the MuJoCo 3D viewer:

```bash
python3 scripts/visualize.py
```

This allows inspection of:

* Conveyor geometry.
* KUKA robot models.
* Package movement.
* Pallet locations.
* Inbound chute.
* Robot workspaces.
* Basic physical interactions.

---

## 18. Training

Start Vectorized PPO training:

```bash
python3 scripts/train.py
```

Training checkpoints are stored in:

```text
checkpoints/
```

The best-performing model is expected at:

```text
checkpoints/best_model.zip
```

---

## 19. Passive Inference

Run the trained policy in the 3D environment:

```bash
python3 scripts/enjoy.py --model checkpoints/best_model.zip
```

This provides a visual representation of the learned palletizing behavior.

---

## 20. Quantitative Benchmark

Run the deterministic benchmark:

```bash
python3 scripts/benchmark.py
```

The benchmark evaluates the trained policy using metrics such as:

* Palletizing success.
* Collision events.
* Cycle time.
* Simulation throughput.

---

## 21. Industrial Relevance

The architecture is intended as a research and simulation framework for robotic logistics applications involving:

* Automated palletizing.
* Parcel sorting.
* Continuous conveyor handling.
* Multi-robot coordination.
* Reinforcement-learning-based manipulation.
* High-throughput fulfillment systems.

The continuous-loop conveyor is particularly useful for studying situations where a package can remain in circulation until an appropriate robot becomes available.

---

## 22. Safety Considerations

This project is a **simulation and research environment**.

The reported collision-free benchmark does not imply that the learned controller is safe for deployment on physical industrial robots.

Before real-world deployment, additional validation would be required, including:

* Hardware safety limits.
* Emergency-stop integration.
* Certified robot safety systems.
* Collision detection independent of the learned policy.
* Workspace and speed limitations.
* Fault handling.
* Sensor validation.
* Real-time controller verification.
* Hardware-in-the-loop testing.
* Formal or empirical safety validation.

The RL policy should therefore be treated as a high-level experimental controller rather than a certified industrial safety system.

---

## 23. Reproducibility

For reproducible experiments, evaluation should record:

* Random seeds.
* MuJoCo version.
* Python version.
* Model checkpoint.
* Environment configuration.
* PPO hyperparameters.
* Number of evaluation episodes.
* Hardware used for benchmarking.

This is particularly important when comparing simulation throughput and RL performance across different machines.

---

## 24. Future Improvements

Potential extensions include:

* Domain randomization for sim-to-real transfer.
* RGB/depth camera observations.
* Vision-based package detection.
* Dynamic pallet layouts.
* Variable package dimensions and weights.
* Conveyor speed randomization.
* More than two robotic agents.
* Curriculum learning.
* Centralized training with decentralized execution (CTDE).
* SAC/TD3 comparison against PPO.
* Real-time trajectory optimization.
* Model predictive safety filtering.
* Hardware-in-the-loop validation.

---

## 25. License

This project is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

## 26. Quick Reference

### Install

```bash
git clone https://github.com/pmalfa31-svg/dual-kuka-cell.git
cd dual-kuka-cell

python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### Visualize

```bash
python3 scripts/visualize.py
```

### Train

```bash
python3 scripts/train.py
```

### Evaluate visually

```bash
python3 scripts/enjoy.py --model checkpoints/best_model.zip
```

### Benchmark

```bash
python3 scripts/benchmark.py
```

---

## Summary

**Dual-KUKA Continuous Palletizing Cell** provides a physics-based simulation framework for studying high-throughput robotic palletizing with two cooperating industrial manipulators.

By combining a continuous conveyor loop, dual KUKA KR6 robots, suction-based package handling, collision-aware coordination, and PPO-based multi-agent reinforcement learning, the project provides a foundation for experimentation in **robotic logistics, multi-agent control, and reinforcement-learning-driven industrial automation**.
