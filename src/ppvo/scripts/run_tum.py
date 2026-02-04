from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import yaml
except ImportError as ex:
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from ex

from ppvo.dataset.tum import TumRgbSequence
from ppvo.system.state import FrameData, SystemState
from ppvo.system.policy import PolicyS1
from ppvo.system.telemetry import Telemetry
from ppvo.system.runner import step  # you implement step(state, policy, cfg, telemetry)


def _R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    # Returns quaternion [x,y,z,w] from rotation matrix.
    # Robust enough for VO outputs.
    m = R.astype(np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # normalize
    n = np.linalg.norm(q) + 1e-12
    return q / n


class TrajectoryVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 5))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)

        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title('3D Trajectory')

        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Z (m)')
        self.ax2.set_title('Top-Down View (X-Z)')
        self.ax2.grid(True)

        self.trajectory_3d = None
        self.trajectory_2d = None
        self.start_marker_3d = None
        self.start_marker_2d = None
        self.current_marker_3d = None
        self.current_marker_2d = None

    def update(self, traj_T_w_c: list[np.ndarray]):
        if len(traj_T_w_c) < 2:
            return

        # Extract positions
        positions = np.array([T[:3, 3] for T in traj_T_w_c])
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Update 3D plot
        self.ax1.clear()
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Y (m)')
        self.ax1.set_zlabel('Z (m)')
        self.ax1.set_title(f'3D Trajectory ({len(traj_T_w_c)} frames)')
        self.ax1.plot(x, y, z, 'b-', linewidth=1.5, alpha=0.7)
        self.ax1.scatter(x[0], y[0], z[0], c='g', s=100, marker='o', label='Start')
        self.ax1.scatter(x[-1], y[-1], z[-1], c='r', s=100, marker='o', label='Current')
        self.ax1.legend()

        # Update 2D top-down view
        self.ax2.clear()
        self.ax2.set_xlabel('X (m)')
        self.ax2.set_ylabel('Z (m)')
        self.ax2.set_title(f'Top-Down View (traveled: {np.linalg.norm(positions[-1] - positions[0]):.2f}m)')
        self.ax2.plot(x, z, 'b-', linewidth=1.5, alpha=0.7)
        self.ax2.scatter(x[0], z[0], c='g', s=100, marker='o', label='Start')
        self.ax2.scatter(x[-1], z[-1], c='r', s=100, marker='o', label='Current')
        self.ax2.grid(True)
        self.ax2.legend()
        self.ax2.axis('equal')

        plt.pause(0.001)

    def close(self):
        plt.ioff()
        plt.show()


def _write_traj_tum(traj_T_w_c: list[np.ndarray], ts_list: list[float], out_path: str) -> None:
    assert len(traj_T_w_c) == len(ts_list)
    with open(out_path, "w", encoding="utf-8") as f:
        for T, ts in zip(traj_T_w_c, ts_list):
            t = T[:3, 3]
            q = _R_to_quat_xyzw(T[:3, :3])  # x y z w
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--tum_dir", type=str, required=True, help="Path to TUM sequence dir, e.g. .../freiburg1_xyz")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--visualize", action="store_true", help="Enable real-time trajectory visualization")
    ap.add_argument("--viz_update_every", type=int, default=10, help="Update visualization every N frames")
    ap.add_argument("--log_every", type=int, default=50, help="Log progress every N frames")
    args = ap.parse_args()

    print(f"[INFO] Loading config: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seq_name = cfg["dataset"]["sequence"]
    out_dir = Path(args.out_dir) / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")

    # Camera intrinsics
    fx = float(cfg["camera"]["fx"])
    fy = float(cfg["camera"]["fy"])
    cx = float(cfg["camera"]["cx"])
    cy = float(cfg["camera"]["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    print(f"[INFO] Loading TUM sequence: {args.tum_dir}")
    seq = TumRgbSequence(args.tum_dir)
    print(f"[INFO] Sequence frames: {len(seq)}")

    state = SystemState(K=K)
    policy = PolicyS1(cfg)
    telemetry = Telemetry()

    # Initialize visualizer if requested
    visualizer = TrajectoryVisualizer() if args.visualize else None

    start = int(cfg["dataset"].get("start", 0))
    step_stride = int(cfg["dataset"].get("step", 1))
    max_frames = cfg["dataset"].get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)

    ts_list: list[float] = []
    frame_count = 0

    print(f"[INFO] Starting loop: start={start} step={step_stride} max_frames={max_frames}")
    for idx, ts, img_gray in seq.iter_gray(start=start, step=step_stride, max_frames=max_frames):
        frame = FrameData(idx=idx, ts=ts, img_gray=img_gray)

        if state.prev is None:
            # Initialize trajectory with identity pose at first frame
            state.prev = frame
            state.traj_T_w_c.append(np.eye(4, dtype=np.float64))
            state.last_T_cur_prev = None
            ts_list.append(ts)
            telemetry.log_frame(idx, {"chosen": {"name": "init", "reason": "INIT"}, "num_matches": 0, "proposals": []})
            frame_count += 1
            continue

        state.cur = frame

        # one step VO (you implement step in system/runner.py)
        step(state, policy, cfg, telemetry)

        ts_list.append(ts)
        frame_count += 1

        # Progress log
        if args.log_every > 0 and (frame_count % args.log_every == 0):
            print(f"[INFO] Frame {frame_count} / {max_frames if max_frames else '?'}")

        # Update visualization
        if visualizer is not None and frame_count % args.viz_update_every == 0:
            visualizer.update(state.traj_T_w_c)
            print(f"[Frame {frame_count}/{max_frames if max_frames else '?'}] Position: {state.traj_T_w_c[-1][:3, 3]}")

        # shift window
        state.prev = state.cur
        state.cur = None

    # Save outputs
    traj_path = str(out_dir / "traj.txt")
    metrics_path = str(out_dir / "metrics.json")
    cfg_path = str(out_dir / "config_used.yaml")

    _write_traj_tum(state.traj_T_w_c, ts_list, traj_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(telemetry.frames, f, indent=2)

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[OK] wrote: {traj_path}")
    print(f"[OK] wrote: {metrics_path}")

    # Keep visualization window open if enabled
    if visualizer is not None:
        print("[INFO] Showing final trajectory. Close the window to exit.")
        visualizer.update(state.traj_T_w_c)
        visualizer.close()


if __name__ == "__main__":
    main()
