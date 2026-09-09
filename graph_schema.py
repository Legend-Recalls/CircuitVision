"""Canonical graph schema v1 — the contract for Stages 2-8.

Nodes: component / pin / net. Edges: component->pin, pin->net.
Never component->component: correction is only possible on pins.
"""
from dataclasses import dataclass, field


@dataclass
class Component:
    id: int
    cls: str
    conf: float
    bbox: tuple  # (x0, y0, x1, y1) pixels
    orientation: int = 0  # 0/90/180/270, 0 = as-drawn default


@dataclass
class Pin:
    id: int
    comp_id: int
    role: str  # p0/p1 for generic 2-T; a/k diode; b/c/e bjt; g/s/d fet; 1 for 1-T
    x: float
    y: float
    source: str = "template"  # template | keypoint | corrected
    conf: float = 1.0


@dataclass
class Net:
    id: int
    pin_ids: list = field(default_factory=list)


@dataclass
class CircuitGraph:
    components: list = field(default_factory=list)
    pins: list = field(default_factory=list)
    nets: list = field(default_factory=list)
    edges_pin_net: list = field(default_factory=list)  # (pin_id, net_id)

    def validate(self):
        comp_ids = {c.id for c in self.components}
        pin_ids = {p.id for p in self.pins}
        assert all(p.comp_id in comp_ids for p in self.pins), "orphan pin"
        assert all(pid in pin_ids for pid, _ in self.edges_pin_net), "dangling edge"
        return True

    def summary(self):
        return (f"{len(self.components)} components, {len(self.pins)} pins, "
                f"{len(self.nets)} nets, {len(self.edges_pin_net)} pin-net edges")


def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def dedup_components(dets, iou_thr=0.5):
    """Merge near-duplicate boxes (keep max conf).

    dets: list of (cls, conf, bbox). Merges same-class overlaps AND
    cross-class stacks (variant confusion: npn/pnp on one transistor).
    Different overlapping components are nearly disjoint in practice
    (max stray IoU observed: 0.45), so 0.5 is safe.
    """
    out = []
    for cls, conf, bbox in sorted(dets, key=lambda d: -d[1]):
        if any(box_iou(bbox, b) > iou_thr for _, _, b in out):
            continue
        out.append((cls, conf, bbox))
    return out
