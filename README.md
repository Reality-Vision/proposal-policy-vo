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
- `vggt`: Vision Geometry Transformer (deep learning-based, optional)
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
│   ├── vggt.py             # VGGT deep learning pose estimation (optional)
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

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker
- Docker Compose (optional, but recommended)

#### Setup with Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/Reality-Vision/proposal-policy-vo
cd proposal-policy-vo
```

2. Download TUM dataset to `./data` directory:
```bash
mkdir -p data
cd data
wget https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
tar -xzf rgbd_dataset_freiburg1_xyz.tgz
cd ..
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

4. Or run with custom parameters:
```bash
docker-compose run --rm ppvo python src/ppvo/scripts/run_tum.py \
    --tum_dir /data/rgbd_dataset_freiburg1_xyz \
    --config src/ppvo/configs/default.yaml \
    --out_dir /outputs
```

#### Setup with Docker only

1. Build the Docker image:
```bash
docker build -t ppvo:latest .
```

2. Run the container (with GPU support):
```bash
docker run --rm \
    --gpus all \
    --runtime=nvidia \
    -v $(pwd)/data:/data \
    -v $(pwd)/outputs:/outputs \
    ppvo:latest \
    python src/ppvo/scripts/run_tum.py \
        --tum_dir /data/rgbd_dataset_freiburg1_xyz \
        --config src/ppvo/configs/default.yaml \
        --out_dir /outputs
```

**Note**: The Docker image is built with NVIDIA CUDA 12.2 support. Remove `--gpus all --runtime=nvidia` if you don't have a GPU.

### Option 2: Local Installation

#### Prerequisites
- Python 3.8+
- miniconda or conda

#### Setup

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

## Development Environment

### VSCode Dev Container (Recommended for Development)

For the best development experience, use VSCode with Dev Containers. This provides a consistent, isolated development environment with all tools pre-configured.

#### Prerequisites
- [Visual Studio Code](https://code.visualstudio.com/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VSCode
- **For GPU support**: [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker) (NVIDIA GPU required)

#### Setup Steps

1. Open the project in VSCode:
```bash
code .
```

2. When prompted, click **"Reopen in Container"**, or:
   - Press `F1` or `Ctrl+Shift+P` (Cmd+Shift+P on Mac)
   - Type "Dev Containers: Reopen in Container"
   - Press Enter

3. VSCode will build the development container (first time takes a few minutes)

4. Once the container is running, you're ready to develop!

#### What's Included

The dev container includes:
- Python 3.10 with all project dependencies
- Development tools: pylint, black, flake8, mypy, pytest
- Debugging tools: gdb, ipdb
- Pre-configured VSCode extensions:
  - Python language support
  - Pylance (intelligent code completion)
  - Black formatter (auto-format on save)
  - Linters (pylint, flake8)
  - Git integration
  - AutoDocstring
- Git and common utilities (zsh, oh-my-zsh)

#### Development Workflow

```bash
# All commands run inside the container

# Run the VO pipeline
python src/ppvo/scripts/run_tum.py --tum_dir /workspace/data/dataset --config src/ppvo/configs/default.yaml

# Run tests (if available)
pytest

# Format code
black src/

# Lint code
pylint src/ppvo/

# Interactive Python
ipython
```

#### Tips
- Your workspace is mounted at `/workspace` in the container
- Bash history is persisted across container rebuilds
- Any changes you make to files are immediately reflected on your host machine
- To rebuild the container: `F1` → "Dev Containers: Rebuild Container"

### GPU Support

This project is now configured with **NVIDIA GPU support** for accelerated computing.

#### Prerequisites for GPU
- NVIDIA GPU with CUDA support
- [NVIDIA Docker runtime](https://github.com/NVIDIA/nvidia-docker) installed
- NVIDIA drivers installed on host machine

#### Verify GPU Access

After opening in the dev container or running with Docker Compose, verify GPU access:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA availability in Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
```

#### What's Included for GPU
- CUDA 12.2 base environment
- PyTorch with CUDA support (pre-installed in dev container)
- CUDA toolkit and libraries
- Automatic GPU device access (`--gpus=all`)

#### Using GPU in Your Code

```python
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move your tensors/models to GPU
# tensor = tensor.to(device)
# model = model.to(device)
```

#### Docker Compose with GPU

The [docker-compose.yml](docker-compose.yml) is already configured for GPU support:

```bash
# Run with GPU
docker-compose up --build

# Or run specific command
docker-compose run --rm ppvo python -c "import torch; print(torch.cuda.is_available())"
```

#### Troubleshooting GPU

If GPU is not detected:
1. Verify NVIDIA drivers: `nvidia-smi` on host
2. Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`
3. Ensure NVIDIA Docker runtime is installed: [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

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

## VGGT Proposal (Optional Deep Learning-Based Pose Estimation)

[VGGT (Vision Geometry Transformer)](https://github.com/facebookresearch/vggt) is a deep learning model for visual odometry that can estimate camera poses directly from RGB image pairs. This is an **optional** proposal source that complements the traditional geometry-based methods.

### Features
- **End-to-end learning**: Trained on large-scale datasets for robust pose estimation
- **GPU acceleration**: Requires CUDA for optimal performance (CPU mode available but slower)
- **Priority in policy**: When enabled, VGGT is prioritized between PnP and essential matrix methods
- **Automatic model download**: Downloads pre-trained weights from HuggingFace on first run

### Installation

#### Option 1: Using pip extras (recommended)
```bash
# Install PyTorch with CUDA support first (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install VGGT as an optional dependency
pip install -e .[vggt]
```

#### Option 2: Direct installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "vggt @ git+https://github.com/facebookresearch/vggt.git"
```

#### Option 3: Using Docker (easiest)
VGGT dependencies are already included in the Docker image. Just enable it in the config.

### Configuration

Enable VGGT in `src/ppvo/configs/default.yaml`:

```yaml
vggt:
  enabled: true                  # Set to false to disable VGGT
  allow_cpu: false               # Set to true to allow CPU inference (slow)
  clear_cache_every: 0           # Clear GPU cache every N frames (0=disabled)
  weights_url: "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
  # weights_path: "/path/to/local/model.pt"  # Optional: use local weights instead
```

### Policy Priority

When VGGT is enabled, the decision policy prioritizes proposals in this order:
1. **PnP** (if available, with quality checks)
2. **VGGT** (if enabled and successful)
3. **Essential Matrix** (geometry-based, with quality checks)
4. **Constant Velocity** (fallback)

### Performance Notes

- **First run**: Downloads ~4GB model weights (cached for future runs)
- **GPU memory**: Requires ~2-3GB GPU memory
- **Speed**: ~100-200ms per frame on modern GPUs (RTX 30/40 series)
- **CPU mode**: 10-50x slower, not recommended for real-time use

### Troubleshooting

If VGGT fails to load:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Test VGGT import
python -c "from vggt.models.vggt import VGGT; print('VGGT OK')"

# Check logs in telemetry for failure reasons (REJECT_VGGT_*)
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
