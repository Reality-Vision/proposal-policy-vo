# src/ppvo/system/runner.py
from __future__ import annotations

import numpy as np

from .state import SystemState
from .telemetry import Telemetry
from .policy import PolicyS1
from ..modules.orb_match import orb_match
from ..modules.emat import propose_emat
from ..modules.const_vel import propose_const_vel
from ..geom.se3 import inv_T


def step(state: SystemState, policy: PolicyS1, cfg: dict, telemetry: Telemetry) -> None:
    """
    One VO step from state.prev -> state.cur.

    Responsibilities:
      1) run perception (feature matching)
      2) generate proposals (const_vel, emat)
      3) ask policy to choose
      4) commit chosen motion to trajectory
      5) log telemetry

    Assumptions:
      - state.prev and state.cur are set
      - state.traj_T_w_c already contains pose for prev frame (at least one entry)
      - transforms follow convention: T_a_b maps points from b to a
        - proposals output T_cur_prev
        - trajectory stores T_w_c
    """
    if state.prev is None or state.cur is None:
        raise ValueError("runner.step requires state.prev and state.cur to be set.")
    if len(state.traj_T_w_c) == 0:
        raise ValueError("state.traj_T_w_c must contain at least the pose for the first frame.")

    # --- 1) Perception: ORB matching
    pts_prev, pts_cur, match_info = orb_match(
        state.prev.img_gray,
        state.cur.img_gray,
        nfeatures=int(cfg["orb"].get("nfeatures", 2000)),
        scaleFactor=float(cfg["orb"].get("scaleFactor", 1.2)),
        nlevels=int(cfg["orb"].get("nlevels", 8)),
        edgeThreshold=int(cfg["orb"].get("edgeThreshold", 31)),
        fastThreshold=int(cfg["orb"].get("fastThreshold", 20)),
        ratio=float(cfg["orb"].get("ratio", 0.8)),
        max_matches=int(cfg["orb"].get("max_matches", 3000)) if cfg["orb"].get("max_matches", 3000) is not None else None,
        mutual_check=bool(cfg["orb"].get("mutual_check", False)),
    )

    # --- 2) Proposals
    proposals = []

    # 2.1 const-vel is always available as a fallback
    prop_cv = propose_const_vel(state.last_T_cur_prev)
    proposals.append(prop_cv)

    # 2.2 emat proposal (only if enough matches)
    min_matches = int(cfg["orb"].get("min_matches", cfg.get("emat", {}).get("min_matches", 120)))
    if pts_prev.shape[0] >= min_matches:
        prop_emat = propose_emat(
            pts_prev,
            pts_cur,
            state.K,
            ransac_thresh_px=float(cfg["emat"].get("ransac_thresh_px", cfg["emat"].get("ransac_thresh_px", 1.0))),
            ransac_prob=float(cfg["emat"].get("ransac_prob", 0.999)),
            min_matches=min_matches,
            recover_pose_min_inliers=int(cfg["emat"].get("recover_pose_min_inliers", cfg["emat"].get("recover_pose_min_inliers", 80))),
        )
        proposals.append(prop_emat)
    else:
        # Make emat "implicitly rejected" visible in logs via const_vel reason later.
        pass

    # --- 3) Decision
    chosen = policy.choose(proposals)

    # --- 4) Commit: update trajectory and motion prior
    # traj stores T_w_c; chosen provides T_cur_prev
    T_w_prev = state.traj_T_w_c[-1]
    T_prev_cur = inv_T(chosen.T_cur_prev)  # because chosen is prev->cur; need cur->prev to compose with T_w_prev? No:
    # T_w_cur = T_w_prev @ T_prev_cur, where T_prev_cur maps cur->prev. Correct given T_a_b maps b->a:
    # - T_w_prev : prev -> w
    # - T_prev_cur : cur -> prev
    # => T_w_prev @ T_prev_cur : cur -> w  (i.e., T_w_c for current)
    T_w_cur = T_w_prev @ T_prev_cur
    state.traj_T_w_c.append(T_w_cur)
    state.last_T_cur_prev = chosen.T_cur_prev.copy()

    # --- 5) Telemetry
    telemetry.log_frame(state.cur.idx, {
        "ts": float(state.cur.ts),
        "match": {
            "num_kp0": int(match_info.get("num_kp0", 0)),
            "num_kp1": int(match_info.get("num_kp1", 0)),
            "num_raw": int(match_info.get("num_raw", 0)),
            "num_good": int(match_info.get("num_good", pts_prev.shape[0])),
        },
        "proposals": [
            {
                "name": p.name,
                "valid": bool(p.valid),
                "reason": str(p.reason),
                "num_inliers": int(p.evidence.num_inliers),
                "inlier_ratio": float(p.evidence.inlier_ratio),
                "reproj_median_px": (None if p.evidence.reproj_median_px is None else float(p.evidence.reproj_median_px)),
            }
            for p in proposals
        ],
        "chosen": {
            "name": chosen.name,
            "reason": chosen.reason,
        },
    })
