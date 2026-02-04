from .proposal import Proposal

class PolicyS1:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def choose(self, proposals: list[Proposal]) -> Proposal:
        # priority: pnp > emat > const_vel
        pnp = next((p for p in proposals if p.name == "pnp" and p.valid), None)
        if pnp and pnp.evidence.num_inliers >= self.cfg["pnp"]["min_inliers"] and \
           (pnp.evidence.reproj_median_px is None or pnp.evidence.reproj_median_px <= self.cfg["pnp"]["max_reproj_median_px"]):
            pnp.reason = "ACCEPT_PNP"
            return pnp

        vggt = next((p for p in proposals if p.name == "vggt" and p.valid), None)
        if vggt:
            vggt.reason = "ACCEPT_VGGT"
            return vggt

        emat = next((p for p in proposals if p.name == "emat" and p.valid), None)
        if emat and emat.evidence.num_inliers >= self.cfg["emat"]["min_inliers"] and \
           emat.evidence.inlier_ratio >= self.cfg["emat"]["min_inlier_ratio"]:
            emat.reason = "ACCEPT_EMAT"
            return emat

        cv = next((p for p in proposals if p.name == "const_vel" and p.valid), None)
        if cv:
            cv.reason = "FALLBACK_CONST_VEL"
            return cv

        # should not happen
        return proposals[0]
