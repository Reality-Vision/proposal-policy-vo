class Telemetry:
    def __init__(self):
        self.frames = []

    def log_frame(self, idx: int, rec: dict):
        rec["frame_idx"] = idx
        self.frames.append(rec)
