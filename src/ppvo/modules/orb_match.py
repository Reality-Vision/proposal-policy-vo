# src/ppvo/modules/orb_match.py
from __future__ import annotations

import cv2
import numpy as np


def orb_match(
    img0_gray_u8: np.ndarray,
    img1_gray_u8: np.ndarray,
    *,
    nfeatures: int = 2000,
    scaleFactor: float = 1.2,
    nlevels: int = 8,
    edgeThreshold: int = 31,
    fastThreshold: int = 20,
    ratio: float = 0.8,
    max_matches: int | None = 3000,
    mutual_check: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    ORB feature matching between two grayscale images.

    Args:
        img0_gray_u8, img1_gray_u8: uint8 grayscale images, shape (H,W)
        ratio: Lowe ratio for knn matching (m.distance < ratio * n.distance)
        mutual_check: if True, enforce symmetric matching (slower, fewer matches)

    Returns:
        pts0: (N,2) float32 pixel coords in img0
        pts1: (N,2) float32 pixel coords in img1
        info: dict with diagnostic info: num_kp0, num_kp1, num_raw, num_good
    """
    if img0_gray_u8 is None or img1_gray_u8 is None:
        raise ValueError("Input images are None")
    if img0_gray_u8.ndim != 2 or img1_gray_u8.ndim != 2:
        raise ValueError("orb_match expects grayscale images (H,W).")

    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels,
        edgeThreshold=edgeThreshold,
        fastThreshold=fastThreshold,
    )

    kp0, des0 = orb.detectAndCompute(img0_gray_u8, None)
    kp1, des1 = orb.detectAndCompute(img1_gray_u8, None)

    info = {
        "num_kp0": 0 if kp0 is None else len(kp0),
        "num_kp1": 0 if kp1 is None else len(kp1),
        "num_raw": 0,
        "num_good": 0,
    }

    if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), info

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _knn_ratio_matches(d0: np.ndarray, d1: np.ndarray):
        knn = bf.knnMatch(d0, d1, k=2)
        good = []
        raw = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            raw += 1
            if m.distance < ratio * n.distance:
                good.append(m)
        return good, raw

    matches01, raw01 = _knn_ratio_matches(des0, des1)
    info["num_raw"] = raw01

    if mutual_check:
        matches10, _ = _knn_ratio_matches(des1, des0)
        # build reverse lookup: trainIdx -> queryIdx for best matches10
        rev = {(m.trainIdx, m.queryIdx) for m in matches10}  # (idx0, idx1)
        matches01 = [m for m in matches01 if (m.queryIdx, m.trainIdx) in rev]

    # sort by distance (smaller is better)
    matches01.sort(key=lambda m: m.distance)

    if max_matches is not None and len(matches01) > max_matches:
        matches01 = matches01[:max_matches]

    pts0 = np.array([kp0[m.queryIdx].pt for m in matches01], dtype=np.float32)
    pts1 = np.array([kp1[m.trainIdx].pt for m in matches01], dtype=np.float32)

    info["num_good"] = int(len(matches01))
    return pts0, pts1, info
