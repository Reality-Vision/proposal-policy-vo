from __future__ import annotations

import os
import tempfile
from typing import Tuple

import numpy as np

from ..system.proposal import Proposal, Evidence
from ..geom.se3 import inv_T

_VGGT_MODEL = None
_VGGT_DEVICE = None
_VGGT_DTYPE = None
_VGGT_CALLS = 0


def _load_vggt_model(cfg: dict):
    global _VGGT_MODEL, _VGGT_DEVICE, _VGGT_DTYPE
    if _VGGT_MODEL is not None:
        return _VGGT_MODEL, _VGGT_DEVICE, _VGGT_DTYPE

    try:
        import torch
        from vggt.models.vggt import VGGT
    except Exception as ex:  # pragma: no cover - environment dependent
        raise ImportError("VGGT is not installed or import failed.") from ex

    allow_cpu = bool(cfg.get("allow_cpu", False))
    if torch.cuda.is_available():
        device = "cuda"
    elif allow_cpu:
        device = "cpu"
    else:
        raise RuntimeError("VGGT requires CUDA. Set vggt.allow_cpu=true to override.")

    # bfloat16 on Ampere+; fallback to float16 otherwise (matches demo_gradio)
    if device == "cuda":
        _VGGT_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        _VGGT_DTYPE = torch.float32

    print("[VGGT] Initializing model...")
    model = VGGT()
    weights_path = cfg.get("weights_path")
    if weights_path:
        print(f"[VGGT] Loading weights from file: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        weights_url = cfg.get("weights_url", "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt")
        print(f"[VGGT] Downloading weights from: {weights_url}")
        model.load_state_dict(torch.hub.load_state_dict_from_url(weights_url))

    print("[VGGT] Model ready.")
    model.eval()
    model = model.to(device)

    _VGGT_MODEL = model
    _VGGT_DEVICE = device
    return _VGGT_MODEL, _VGGT_DEVICE, _VGGT_DTYPE


def _write_temp_pair(img0_gray: np.ndarray, img1_gray: np.ndarray) -> Tuple[str, list[str]]:
    import cv2

    tmp_dir = tempfile.mkdtemp(prefix="ppvo_vggt_")
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)

    path0 = os.path.join(img_dir, "000000.png")
    path1 = os.path.join(img_dir, "000001.png")
    cv2.imwrite(path0, img0)
    cv2.imwrite(path1, img1)

    return tmp_dir, [path0, path1]


def _extrinsic_to_T(extrinsic: np.ndarray) -> np.ndarray:
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = extrinsic.astype(np.float64)
        return T
    raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}")


def propose_vggt(img_prev_gray: np.ndarray, img_cur_gray: np.ndarray, cfg: dict) -> Proposal:
    """
    Estimate relative pose using VGGT from two RGB frames.

    Returns:
        Proposal with name "vggt" and T_cur_prev (4x4).
        If failed, valid=False and T_cur_prev=I.
    """
    I = np.eye(4, dtype=np.float64)
    ev = Evidence(num_inliers=0, inlier_ratio=0.0, reproj_median_px=None)

    global _VGGT_CALLS
    try:
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as ex:  # pragma: no cover - environment dependent
        return Proposal("vggt", I, ev, valid=False, reason=f"REJECT_VGGT_IMPORT:{ex}")

    try:
        model, device, dtype = _load_vggt_model(cfg)
    except Exception as ex:
        return Proposal("vggt", I, ev, valid=False, reason=f"REJECT_VGGT_MODEL:{ex}")

    tmp_dir = None
    try:
        tmp_dir, image_paths = _write_temp_pair(img_prev_gray, img_cur_gray)
        images = load_and_preprocess_images(image_paths).to(device)

        from contextlib import nullcontext
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if device == "cuda" else nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                predictions = model(images)

        extrinsic, _intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        extrinsic = extrinsic.cpu().numpy().squeeze(0)  # (S, 3, 4) or (S, 4, 4)

        if extrinsic.shape[0] < 2:
            return Proposal("vggt", I, ev, valid=False, reason="REJECT_VGGT_TOO_FEW_VIEWS")

        T0 = _extrinsic_to_T(extrinsic[0])
        T1 = _extrinsic_to_T(extrinsic[1])

        T_cur_prev = T1 @ inv_T(T0)
        return Proposal("vggt", T_cur_prev, ev, valid=True, reason="VGGT_OK")
    except Exception as ex:
        return Proposal("vggt", I, ev, valid=False, reason=f"REJECT_VGGT_RUN:{ex}")
    finally:
        _VGGT_CALLS += 1
        if tmp_dir is not None:
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        try:
            clear_every = int(cfg.get("clear_cache_every", 0) or 0)
            if device == "cuda" and clear_every > 0 and (_VGGT_CALLS % clear_every == 0):
                torch.cuda.empty_cache()
        except Exception:
            pass
