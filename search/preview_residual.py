"""Opt-in bounded preview correction for the rain team's difficult matchups.

An offline predictor is never a promotion decision. The candidate only reranks
plans near the incumbent prediction, and only on its declared matchup groups.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from battle_state import to_id
from search.human.preview_model import PLANS
from search.human.preview_value import PreviewValue
from search.tracker import Tracker
from search.preview_guard import DEFAULT_MODEL, ROOT

TARGET = ('garchomp', 'politoed', 'incineroar', 'archaludon', 'farigiraf', 'golisopod')
RISKS = ('kingambit', 'milotic', 'annihilape', 'gardevoir', 'sylveon', 'floetteeternal')
VERSION = 1


def features(own, opp, plan):
    own, opp = [to_id(x) for x in own], set(map(to_id, opp))
    lead = {own[i] for i in plan[:2]}
    brought = {own[i] for i in plan}
    six = np.array([s in own for s in TARGET], dtype=np.float32)
    l = np.array([s in lead for s in TARGET], dtype=np.float32)
    b = np.array([s in brought for s in TARGET], dtype=np.float32)
    r = np.array([s in opp for s in RISKS], dtype=np.float32)
    return np.concatenate([six, b, l, r, np.outer(l, r).ravel(), np.outer(b, r).ravel()])


def bounded_choice(scores, corrections, legal_indices, max_drop=.025):
    eligible = [i for i in legal_indices if np.isfinite(scores[i]) and np.isfinite(corrections[i])]
    if not eligible:
        return None
    best = max(scores[i] for i in eligible)
    eligible = [i for i in eligible if scores[i] >= best - max_drop]
    logits = np.log(np.clip(scores, 1e-6, 1-1e-6) / np.clip(1-scores, 1e-6, 1))
    return max(eligible, key=lambda i: logits[i] + corrections[i])


class PreviewResidual:
    def __init__(self, config):
        self.max_drop = float(config.get('max_drop', .025))
        if not 0 <= self.max_drop <= .025:
            raise ValueError('preview residual may not widen the incumbent eligibility bound')
        path = ROOT / config['path']
        blob = json.loads(path.read_text())
        self.cap = float(blob['cap'])
        if blob.get('version') != VERSION or not 0 < self.cap <= .35:
            raise ValueError('unsupported preview residual model')
        baseline = ROOT / config.get('baseline', DEFAULT_MODEL)
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != blob['baseline_sha256']:
            raise ValueError('preview residual baseline checksum differs')
        self.baseline = PreviewValue(baseline)
        self.layers = [(np.array(l['W']), np.array(l['b'])) for l in blob['layers']]
        dim = len(features(TARGET, RISKS, PLANS[0]))
        for w, b in self.layers:
            if w.ndim != 2 or w.shape[1] != dim or b.shape != (w.shape[0],):
                raise ValueError('invalid residual feature dimensions')
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('nonfinite residual weights')
            dim = len(b)
        if dim != 1:
            raise ValueError('residual must predict one correction')

    def choose(self, ctx):
        request = ctx['request']
        own = [p['details'].split(',')[0] for p in request['side']['pokemon']]
        tracker = Tracker(request['side']['id']).feed(ctx.get('public_log') or [])
        opp = tracker.preview[tracker.foe]
        if (len(own) != 6 or set(map(to_id, own)) != set(TARGET) or len(opp) != 6
                or not set(map(to_id, opp)) & set(RISKS)):
            return None
        h = np.stack([features(own, opp, p) for p in PLANS])
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers)-1:
                h = np.maximum(h, 0)
        delta = self.cap * np.tanh(h[:, 0])
        choices = ['team ' + ','.join(str(j+1) for j in p) for p in PLANS]
        legal = set(ctx['choices'])
        k = bounded_choice(self.baseline.probs(own, opp), delta,
                           [i for i, c in enumerate(choices) if c in legal], self.max_drop)
        return choices[k] if k is not None else None
