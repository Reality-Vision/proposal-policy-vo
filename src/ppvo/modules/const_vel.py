import numpy as np
from ..system.proposal import Proposal, Evidence

def propose_const_vel(last_T_cur_prev: np.ndarray | None) -> Proposal:
    T = np.eye(4) if last_T_cur_prev is None else last_T_cur_prev.copy()
    return Proposal("const_vel", T, Evidence(), valid=True)
