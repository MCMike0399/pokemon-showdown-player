"""Opt-in, bounded reranking of the incumbent preview around Intimidate risks.

This is an experimental prior, not a claim that avoiding Incineroar wins games.
Only public opposing species and our own known abilities enter the adjustment.
"""
from pathlib import Path

import numpy as np

from battle_state import to_id
from search.human.preview_model import PLANS
from search.human.preview_value import PreviewValue
from search.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = 'search/human/models/pvalue-661448b6a320fce0.json'
RISKS = frozenset({'kingambit', 'milotic', 'annihilape'})


def rerank(scores, abilities, opposing, legal, max_drop=.025, penalty=.04):
    """Keep plans within max_drop of the best legal prediction, then penalize risky leads."""
    allowed = set(legal)
    candidates = [(i, 'team ' + ','.join(str(j + 1) for j in p)) for i, p in enumerate(PLANS)]
    candidates = [(i, choice) for i, choice in candidates if choice in allowed and np.isfinite(scores[i])]
    if not candidates:
        return None
    best = max(float(scores[i]) for i, _ in candidates)
    risk = min(1., len(set(map(to_id, opposing)) & RISKS) / 2)
    eligible = [(i, c) for i, c in candidates if scores[i] >= best - max_drop]
    def value(item):
        i, _ = item
        leads = sum(to_id(abilities[j]) == 'intimidate' for j in PLANS[i][:2])
        return float(scores[i]) - penalty * risk * leads
    return max(eligible, key=value)[1]


class PreviewGuard:
    def __init__(self, config):
        self.max_drop = float(config.get('max_drop', .025))
        self.penalty = float(config.get('penalty', .04))
        if not 0 <= self.max_drop <= .1 or not 0 <= self.penalty <= .2:
            raise ValueError('preview guard adjustments must be bounded')
        self.model = PreviewValue(ROOT / config.get('path', DEFAULT_MODEL))

    def choose(self, ctx):
        req = ctx['request']
        party = req['side']['pokemon']
        tracker = Tracker(req['side']['id']).feed(ctx.get('public_log') or [])
        opposing = tracker.preview.get(tracker.foe) or []
        if len(party) != 6 or len(opposing) < 4:
            return None
        own = [p['details'].split(',')[0] for p in party]
        abilities = [p.get('baseAbility') or p.get('ability') or '' for p in party]
        return rerank(self.model.probs(own, opposing), abilities, opposing, ctx['choices'],
                      self.max_drop, self.penalty)
