# src/ppvo/modules/triangulate.py
from __future__ import annotations
import numpy as np
import cv2

def triangulate_two_view(
    pts_prev: np.ndarray,   # (N,2) pixels
    pts_cur: np.ndarray,    # (N,2) pixels
    K: np.ndarray,          # (3,3)
    T_cur_prev: np.ndarray, # (4,4) prev->cur
    *,
    max_reproj_median_px: float = 2.5,
    min_points: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_prev: (M,3) 3D points in prev camera frame
      keep_idx: (M,) indices into input correspondences that survived filtering
      reproj_err_px: (M,) reprojection error in prev (or combined) for debugging
    """
    if pts_prev.shape[0] < min_points:
        return np.zeros((0,3), np.float64), np.zeros((0,), np.int32), np.zeros((0,), np.float64)

    K64 = np.asarray(K, dtype=np.float64)
    p0 = np.asarray(pts_prev, dtype=np.float64)
    p1 = np.asarray(pts_cur, dtype=np.float64)

    R = T_cur_prev[:3, :3]
    t = T_cur_prev[:3, 3].reshape(3, 1)

    P0 = K64 @ np.hstack([np.eye(3), np.zeros((3,1))])     # prev cam
    P1 = K64 @ np.hstack([R, t])                           # cur cam

    # cv2.triangulatePoints expects 2xN
    X_h = cv2.triangulatePoints(P0, P1, p0.T, p1.T)        # 4xN
    X_prev = (X_h[:3, :] / (X_h[3:4, :] + 1e-12)).T        # Nx3

    # Cheirality: depth > 0 in both cameras
    z0 = X_prev[:, 2]
    X_cur = (R @ X_prev.T + t).T
    z1 = X_cur[:, 2]
    mask_front = (z0 > 1e-6) & (z1 > 1e-6)

    # Reprojection error (median-based filtering, robust)
    x0_proj = (P0 @ np.hstack([X_prev, np.ones((X_prev.shape[0],1))]).T).T
    x0_proj = x0_proj[:, :2] / (x0_proj[:, 2:3] + 1e-12)
    e0 = np.linalg.norm(x0_proj - p0, axis=1)

    x1_proj = (P1 @ np.hstack([X_prev, np.ones((X_prev.shape[0],1))]).T).T
    x1_proj = x1_proj[:, :2] / (x1_proj[:, 2:3] + 1e-12)
    e1 = np.linalg.norm(x1_proj - p1, axis=1)

    e = 0.5 * (e0 + e1)
    mask = mask_front

    # Optional robust reprojection gating
    if np.any(mask):
        med = np.median(e[mask])
        # keep if error not too high relative to median, and below an absolute threshold
        mask = mask & (e <= max(max_reproj_median_px, 3.0 * med))

    keep_idx = np.nonzero(mask)[0].astype(np.int32)
    return X_prev[keep_idx], keep_idx, e[keep_idx]
