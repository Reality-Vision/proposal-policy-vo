from dataclasses import dataclass
import numpy as np

@dataclass
class Evidence:
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reproj_median_px: float | None = None

@dataclass
class Proposal:
    name: str
    T_cur_prev: np.ndarray  # 4x4
    evidence: Evidence
    valid: bool = True
    reason: str = ""
