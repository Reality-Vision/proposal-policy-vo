# proposal-policy-vo

A small **educational / research reference implementation** of a **proposal → policy → commit** Visual Odometry (VO) pipeline in Python.

This repo is intentionally minimal: it emphasizes **system-centric orchestration** (multiple pose proposals + explicit decision policy + telemetry), not production performance.

---

## What this repo is (and is not)

**This repo is:**
- A runnable monocular VO demo with a **Proposal / Evidence / Policy** architecture
- A clean starting point for adding more proposal sources later (e.g., depth-based VO, ICP refine)

**This repo is NOT:**
- A full SLAM system (no loop closing, relocalization, global BA, multi-session, etc.)
- A “Python version” or “subset” of any larger private system

---

## Core idea

For each frame pair *(prev → cur)*, the system generates multiple pose **proposals** (candidates), each with **evidence**. A **policy** chooses which proposal to commit.

Example proposal sources:
- `emat`: Essential matrix + `recoverPose` (geometry-only)
- `const_vel`: constant-velocity fallback (motion prior)

The chosen proposal is committed to the trajectory, and every decision is logged in telemetry for debugging and analysis.

---

## Repository structure

```
ppvo/
├── dataset/
│   └── tum.py              # TUM RGB-D dataset loader
├── geom/
│   ├── se3.py              # SE(3) utilities
│   └── project.py          # Projection utilities
├── modules/
│   ├── orb_match.py        # ORB feature matching
│   ├── emat.py             # Essential matrix estimation
│   ├── triangulate.py      # Triangulation
│   └── const_vel.py        # Constant velocity model
├── system/
│   ├── state.py            # System state management
│   ├── proposal.py         # Proposal definitions
│   ├── policy.py           # Decision policy
│   ├── runner.py           # Main VO step logic
│   └── telemetry.py        # Logging and metrics
├── scripts/
│   └── run_tum.py          # Run VO on TUM dataset
└── configs/
    └── default.yaml        # Default configuration
```

---

## Installation

### Prerequisites
- Python 3.8+
- miniconda or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Reality-Vision/proposal-policy-vo
cd ppvo
```

2. Install in editable mode:
```bash
pip install -e .
```

This will install the package with its dependencies:
- numpy
- opencv-python
- pyyaml
- matplotlib (for visualization)

---

## Quick Start

### 1. Download TUM dataset

```bash
# Example: freiburg1_xyz sequence
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
```

### 2. Run Visual Odometry

**Basic run:**
```bash
python src/ppvo/scripts/run_tum.py \
    --tum_dir /path/to/rgbd_dataset_freiburg1_xyz \
    --config src/ppvo/configs/default.yaml \
    --out_dir outputs
```

**With real-time visualization:**
```bash
python src/ppvo/scripts/run_tum.py \
    --tum_dir /path/to/rgbd_dataset_freiburg1_xyz \
    --config src/ppvo/configs/default.yaml \
    --visualize \
    --viz_update_every 20
```

### 3. View results

The script generates:
- `outputs/{sequence}/traj.txt` - Estimated camera trajectory (TUM format)
- `outputs/{sequence}/metrics.json` - Per-frame telemetry and decisions
- `outputs/{sequence}/config_used.yaml` - Configuration snapshot

---

## Configuration

Edit [src/ppvo/configs/default.yaml](src/ppvo/configs/default.yaml) to customize:

- **Dataset settings**: sequence name, frame range, step size
- **Camera intrinsics**: fx, fy, cx, cy
- **ORB parameters**: feature count, scale factor, thresholds
- **Essential matrix RANSAC**: inlier thresholds, min inliers
- **Policy preferences**: proposal priority order

Example:
```yaml
dataset:
  sequence: freiburg1_xyz
  max_frames: 2000
  step: 1

policy:
  prefer: ["emat", "const_vel"]

orb:
  nfeatures: 2000
  ratio: 0.8
```

---

## Command-line Arguments

```
--tum_dir         Path to TUM sequence directory (required)
--config          Config YAML file (default: configs/default.yaml)
--out_dir         Output directory (default: outputs)
--visualize       Enable real-time 3D trajectory visualization
--viz_update_every  Update visualization every N frames (default: 10)
```

---

## Visualization

When `--visualize` is enabled, you'll see:
- **Left panel**: 3D trajectory view (can be rotated)
- **Right panel**: Top-down view (X-Z plane)
- **Green marker**: Start position
- **Red marker**: Current position
- **Blue line**: Camera trajectory

The window stays open after completion. Close it to exit.

---

## Extending the system

To add a new proposal source:

1. Create a new module in `ppvo/modules/` (e.g., `icp.py`)
2. Implement your estimator function
3. Register it in `system/runner.py` to generate proposals
4. Add it to the policy preference list in `configs/default.yaml`

Example:
```python
# In system/runner.py
from ppvo.modules.icp import estimate_icp

def step(state, policy, cfg, telemetry):
    proposals = []

    # Existing proposals
    proposals.append(try_emat(...))
    proposals.append(try_const_vel(...))

    # New proposal
    proposals.append(try_icp(...))  # Your new method

    chosen = policy.choose(proposals)
    state.commit(chosen)
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- TUM RGB-D Dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset
- OpenCV for computer vision primitives
