from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2
import numpy as np


@dataclass
class TumRgbEntry:
    ts: float
    path: str


def _read_rgb_txt(rgb_txt_path: str) -> List[TumRgbEntry]:
    entries: List[TumRgbEntry] = []
    base = os.path.dirname(rgb_txt_path)

    with open(rgb_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            ts = float(parts[0])
            rel = parts[1]
            entries.append(TumRgbEntry(ts=ts, path=os.path.join(base, rel)))
    return entries


class TumRgbSequence:
    def __init__(self, seq_dir: str):
        self.seq_dir = seq_dir
        rgb_txt = os.path.join(seq_dir, "rgb.txt")
        if not os.path.isfile(rgb_txt):
            raise FileNotFoundError(f"Missing rgb.txt: {rgb_txt}")
        self.entries = _read_rgb_txt(rgb_txt)

    def __len__(self) -> int:
        return len(self.entries)

    def iter_gray(
        self,
        *,
        start: int = 0,
        step: int = 1,
        max_frames: int | None = None,
    ) -> Iterator[Tuple[int, float, np.ndarray]]:
        end = len(self.entries) if max_frames is None else min(len(self.entries), start + max_frames * step)
        idx = 0
        for i in range(start, end, step):
            e = self.entries[i]
            img = cv2.imread(e.path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {e.path}")
            yield idx, e.ts, img
            idx += 1
