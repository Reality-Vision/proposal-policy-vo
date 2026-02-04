# src/ppvo/modules/emat.py
from __future__ import annotations

import cv2
import numpy as np

from ..system.proposal import Proposal, Evidence
from ..geom.se3 import Rt_to_T


def propose_emat(
    pts_prev: np.ndarray,
    pts_cur: np.ndarray,
    K: np.ndarray,
    *,
    ransac_thresh_px: float = 1.0,
    ransac_prob: float = 0.999,
    min_matches: int = 120,
    recover_pose_min_inliers: int = 80,
) -> Proposal:
    """
    Estimate relative pose using Essential matrix (monocular, up-to-scale).

    Args:
        pts_prev, pts_cur: (N,2) float pixel coordinates.
        K: (3,3) camera intrinsics.
        ransac_thresh_px: RANSAC reprojection threshold in pixels (OpenCV uses pixel distance).
        ransac_prob: RANSAC confidence.
        min_matches: minimum number of correspondences required to attempt E.
        recover_pose_min_inliers: minimum inliers required after recoverPose.

    Returns:
        Proposal with name "emat" and T_cur_prev (4x4).
        If failed, valid=False and T_cur_prev=I.
    """
    I = np.eye(4, dtype=np.float64)
    ev = Evidence(num_inliers=0, inlier_ratio=0.0, reproj_median_px=None)

    if pts_prev is None or pts_cur is None:
        return Proposal("emat", I, ev, valid=False, reason="REJECT_EMAT_NO_POINTS")
    if pts_prev.shape[0] != pts_cur.shape[0] or pts_prev.shape[0] < min_matches:
        return Proposal(
            "emat",
            I,
            ev,
            valid=False,
            reason=f"REJECT_EMAT_TOO_FEW_MATCHES:{int(0 if pts_prev is None else pts_prev.shape[0])}",
        )

    # Ensure float64 for OpenCV numerical stability
    p0 = np.asarray(pts_prev, dtype=np.float64)
    p1 = np.asarray(pts_cur, dtype=np.float64)
    K64 = np.asarray(K, dtype=np.float64)

    # findEssentialMat expects points in pixels when K is provided.
    E, mask = cv2.findEssentialMat(
        p0,
        p1,
        cameraMatrix=K64,
        method=cv2.RANSAC,
        prob=ransac_prob,
        threshold=ransac_thresh_px,
    )

    if E is None or mask is None:
        return Proposal("emat", I, ev, valid=False, reason="REJECT_EMAT_FIND_E_FAILED")

    mask = mask.reshape(-1).astype(bool)
    num_inliers = int(mask.sum())
    inlier_ratio = float(num_inliers) / float(p0.shape[0] + 1e-9)
    ev.num_inliers = num_inliers
    ev.inlier_ratio = inlier_ratio

    if num_inliers < recover_pose_min_inliers:
        return Proposal(
            "emat",
            I,
            ev,
            valid=False,
            reason=f"REJECT_EMAT_TOO_FEW_INLIERS:{num_inliers}",
        )

    # E could be 3x3 or 3x3*k (stack) if multiple solutions are returned.
    # recoverPose expects a single 3x3 E. If multiple, pick the first block.
    if E.shape[0] > 3 or E.shape[1] > 3:
        E = E[:3, :3]

    # recoverPose returns R, t (unit norm), and an updated inlier mask
    # Note: recoverPose can further reduce inliers.
    retval, R, t, mask_pose = cv2.recoverPose(E, p0, p1, cameraMatrix=K64, mask=mask.astype(np.uint8))

    if retval is None or int(retval) <= 0:
        return Proposal("emat", I, ev, valid=False, reason="REJECT_EMAT_RECOVERPOSE_FAILED")

    num_inliers_pose = int(retval)
    # Update evidence with post-recoverPose inliers
    ev.num_inliers = num_inliers_pose
    ev.inlier_ratio = float(num_inliers_pose) / float(p0.shape[0] + 1e-9)

    if num_inliers_pose < recover_pose_min_inliers:
        return Proposal(
            "emat",
            I,
            ev,
            valid=False,
            reason=f"REJECT_EMAT_RECOVERPOSE_TOO_FEW_INLIERS:{num_inliers_pose}",
        )

    T_cur_prev = Rt_to_T(R, t.reshape(3))
    return Proposal("emat", T_cur_prev, ev, valid=True, reason="EMAT_OK")
