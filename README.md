# Multi-Modal Whole-Body Control

**Multi-Modal Whole-Body Control** is a research-oriented framework for learning **whole-body control policies** for humanoid and general articulated robots from **heterogeneous motion signals**.  
Built on **NVIDIA Isaac Sim / Isaac Lab** and **RSL-RL**, the framework targets **large-scale parallel simulation** and **multi-modal imitation learning**, with a focus on *robust motion tracking* and *cross-modal embodiment*.

The core philosophy of this repository is to treat **whole-body control as a multi-modal sequence alignment problem**:  
a robot policy learns to coordinate its full-body dynamics by jointly conditioning on robot-centric states and external motion descriptors such as **human body pose (SMPL-X)** and **SE(3) keypoints**.

The framework currently focuses on **Unitree G1**, but is designed to be extensible to other humanoid or whole-body platforms.

---

## Key Ideas

- **Whole-body control at scale**  
  Designed for thousands of parallel environments in Isaac Lab, enabling fast and stable on-policy learning.

- **Multi-modal motion supervision**  
  Policies can be conditioned on:
  - robot joint trajectories,
  - human SMPL-X full-body motion,
  - 3D / SE(3) keypoint trajectories.

- **Unified tracking & imitation**  
  Motion tracking, multi-motion training, and GAE-Mimic–style imitation are expressed within a single task framework.

- **Data-driven curricula**  
  Motion sampling ratios adapt automatically based on tracking performance.

---

## Features

- **Isaac Lab task definitions** for:
  - single-motion tracking,
  - multi-motion tracking,
  - multi-modal imitation (GAE-Mimic style).

- **Whole-body robot models**
  - Full articulation and actuator configuration (Unitree G1).
  - Per-joint-group stiffness, damping, and action scaling.

- **Multi-modal motion datasets**
  - NPZ-based motion clips with lazy loading.
  - Optional extensions with SMPL-X and robot keypoints.

- **High-performance dataloaders**
  - Vectorized global indexing over concatenated motion sequences.
  - Weighted sampling and curriculum support.
  - Distributed sharding for multi-GPU training.

- **RSL-RL integration**
  - PPO-style training at scale.
  - Logging, checkpointing, video recording, and resume support.

---

## Repository Structure

```

.
├── source/whole_body_control
│   ├── robots/              # Robot models and actuators (e.g., Unitree G1)
│   ├── tasks/               # Tracking / Multi-Tracking / GAEMimic tasks
│   ├── utils/               # Motion datasets, dataloaders, runners
│
├── scripts
│   ├── rsl_rl/
│   │   ├── train.py         # Training entry point
│   │   └── play.py          # Policy rollout (if available)
│   ├── tools/               # Env listing, random / zero agents
│   └── data/                # Dataset preprocessing utilities
│
├── third_party/rsl_rl       # Vendorized RSL-RL
├── docs/env_setup.md        # Detailed environment setup
└── README.md

```

---

## Tasks and Environments

All environments are implemented using **Isaac Lab task abstractions**.

### Motion Tracking

- **`TrackingEnvCfg`**
  - Single reference motion tracking
  - Joint position control
  - Dense rewards on pose, velocity, orientation, contacts
  - Privileged observations for the critic

### Multi-Motion Tracking

- **`MultiTracking_TrackingEnvCfg`**
  - Samples from multiple motion clips
  - Dataset-driven command interface
  - Performance-based curriculum over motion difficulty

### Multi-Modal Imitation (GAE-Mimic)

- **`GAEMimic_TrackingEnvCfg`**
  - Conditions policy on:
    - robot joint trajectories,
    - SMPL-X human body pose,
    - SE(3) keypoint trajectories
  - Designed for cross-modal imitation and embodiment transfer

---

## Robot Model

### Unitree G1

Defined in `whole_body_control/robots/g1.py` using:

- `ArticulationCfg`
- `ImplicitActuatorCfg`

Includes:
- Full-body joint limits and initial posture
- Per-joint-group actuator stiffness and damping
- Action scaling for stable RL control

---

## Motion Datasets

<details>
<summary><strong>Click to expand dataset details</strong></summary>

### Dataset Layout

```

datasets/
└── extended_datasets/
└── 100style_dataset/
├── g1/
│   ├── *.npz
└── info.yaml

````

Each dataset provides:
- Preprocessed motion clips in NPZ format
- `info.yaml` describing splits and motion quality

---

### Dataset APIs

- **`Motion_Dataset`**
  - Lazy loading of motion clips
  - Robot joint and body trajectories with metadata

- **`Unify_Motion_Dataset`**
  - Extends with SMPL-X body pose and robot keypoints
  - Flattens multi-modal data for RL observations

- **`Motion_Dataloader` / `Unify_Motion_Dataloader`**
  - Concatenated tensors with efficient global indexing
  - Weighted sampling and distributed sharding

</details>

---

## Environment Setup

For full instructions, see [`docs/env_setup.md`](docs/env_setup.md).

### Quick Summary

<details>
<summary><strong>Click to expand environment setup</strong></summary>

```bash
conda create -n env_mimic python=3.10
conda activate env_mimic

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

pip install 'isaacsim[all,extscache]==4.5.0' \
  --extra-index-url https://pypi.nvidia.com
````

</details>

---

## Installation

<details>
<summary><strong>Click to expand installation steps</strong></summary>

```bash
# Install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 90b79bb2d44feb8d833f260f2bf37da3487180ba
./isaaclab.sh -i

# (Optional) install RSL-RL
./isaaclab.sh -p -m pip install -e path/to/rsl_rl

# Install this repository
cd source/whole_body_control
pip install -e .
```

</details>

---

## Dataset Download

<details>
<summary><strong>Click to expand dataset download</strong></summary>

Preprocessed datasets (with SMPL-X and keypoints) are available at:

* [https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset](https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset)

Unzip under the `datasets/` directory.

</details>

---

## Quick Start

<details>
<summary><strong>Click to expand quick start</strong></summary>

### List Available Environments

```bash
python scripts/tools/list_envs.py
```

Example task ID:

```
MultiTracking-Flat-G1-v0
```

---

### Train a Policy

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task MultiTracking-Flat-G1-v0
```

Common options:

* `--num_envs`
* `--seed`
* `--max_iterations`
* `--video`, `--video_interval`
* `--distributed` (multi-GPU)

</details>

---

## Typical Workflow

<details>
<summary><strong>Click to expand workflow</strong></summary>

1. Prepare or extend motion datasets (`scripts/data/`)
2. Select task type and modality
3. Train whole-body policies with RSL-RL
4. Evaluate stability, contacts, and motion fidelity

</details>

---

## License

* Isaac Lab components: **BSD-3-Clause**
* RSL-RL: see `third_party/rsl_rl`
* Robot assets (e.g., Unitree G1): subject to original licenses
