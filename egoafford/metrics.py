"""Pixel-accumulated cIoU and sample-averaged fixed-role diagnostics.

Scores are fractions, not percentages. No role reassignment or GT matching.
"""
import math
import numpy as np
from . import ROLES


class Metrics:
    def __init__(self):
        self.n = 0
        self.intersection = [0] * 3
        self.union = [0] * 3
        self.iou = [0.] * 3
        self.nonempty_iou = [0.] * 3
        self.gt_present = [0] * 3
        self.tp = [0] * 3
        self.fp = [0] * 3
        self.fn = [0] * 3

    def update(self, prediction, ground_truth):
        p, g = np.asarray(prediction), np.asarray(ground_truth)
        if p.dtype != np.bool_ or g.dtype != np.bool_:
            raise ValueError('Metrics requires boolean masks; binarize explicitly before evaluation.')
        if p.shape != g.shape or p.ndim != 3 or p.shape[0] != 3 or 0 in p.shape:
            raise ValueError('Expected equal non-empty (3, H, W) arrays in O/I/D order.')
        self.update_counts((p & g).sum((1, 2)).tolist(), (p | g).sum((1, 2)).tolist(),
                           g.any((1, 2)).tolist(), p.any((1, 2)).tolist())

    def update_counts(self, intersection, union, gt_present, pred_present):
        vectors = [intersection, union, gt_present, pred_present]
        if any(len(v) != 3 for v in vectors):
            raise ValueError('Every count/presence vector must contain exactly three roles.')
        for k in range(3):
            i, u = intersection[k], union[k]
            if isinstance(i, bool) or isinstance(u, bool) or not isinstance(i, int) or not isinstance(u, int):
                raise ValueError('Pixel counts must be integers.')
            if not 0 <= i <= u:
                raise ValueError('Expected 0 <= intersection <= union.')
            gp, pp = gt_present[k], pred_present[k]
            if type(gp) is not bool or type(pp) is not bool:
                raise ValueError('Presence flags must be booleans.')
            if (u == 0) != (not gp and not pp) or (i > 0 and not (gp and pp)):
                raise ValueError('Pixel counts contradict role presence.')
            score = i / u if u else 1.
            self.intersection[k] += i
            self.union[k] += u
            self.iou[k] += score
            if gp:
                self.gt_present[k] += 1
                self.nonempty_iou[k] += score
            self.tp[k] += int(gp and pp)
            self.fp[k] += int(pp and not gp)
            self.fn[k] += int(gp and not pp)
        self.n += 1

    def compute(self):
        if not self.n:
            raise ValueError('Cannot evaluate an empty dataset.')
        per_role = {}
        f1s = []
        for k, role in enumerate(ROLES):
            den = 2 * self.tp[k] + self.fp[k] + self.fn[k]
            f1 = 2 * self.tp[k] / den if den else None
            if f1 is not None:
                f1s.append(f1)
            per_role[role] = dict(NE_mIoU=self.nonempty_iou[k] / self.gt_present[k] if self.gt_present[k] else None,
                                  gt_present=self.gt_present[k], TP=self.tp[k], FP=self.fp[k], FN=self.fn[k], presence_F1=f1)
        den = 2 * sum(self.tp) + sum(self.fp) + sum(self.fn)
        total_union = sum(self.union)
        return dict(num_samples=self.n, role_order=list(ROLES),
                    cIoU=sum(self.intersection) / total_union if total_union else 1.,
                    gIoU=sum(self.iou) / (3 * self.n), component_wise=per_role,
                    presence_macro_F1=sum(f1s) / len(f1s) if f1s else None,
                    presence_micro_F1=2 * sum(self.tp) / den if den else None)
