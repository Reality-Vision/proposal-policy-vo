from dataclasses import dataclass, field
import numpy as np

@dataclass
class FrameData:
    idx: int
    ts: float
    img_gray: np.ndarray

@dataclass
class SystemState:
    K: np.ndarray  # 3x3
    prev: FrameData | None = None
    cur: FrameData | None = None

    traj_T_w_c: list[np.ndarray] = field(default_factory=list)
    last_T_cur_prev: np.ndarray | None = None  # motion prior

    # VO最小map：map points + descriptors
    map_X_w: np.ndarray | None = None         # (M,3)
    map_desc: np.ndarray | None = None        # (M,32) ORB desc

    cache: dict = field(default_factory=dict)
