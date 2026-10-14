"""Campaign c3 operations: settled gate summaries, and promotion of the human opponent policy.

    .venv/bin/python search/human/campaign.py status
    .venv/bin/python search/human/campaign.py promote [--force]

`status` prints each stage's summary (search/value_gate.summarize with the protocol's
pairs) and whether the protocol's promotion condition holds. `promote` refuses unless
dev and confirm are complete and passed, the supplementary standard gate (if present)
does not block (losses > gains with p <= 0.05), and the candidate file's SHA matches
the protocol; it then writes the team's opp-current.json pointer and logs the campaign.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from search.value_gate import summarize  # noqa: E402

CAMP = ROOT / 'data' / 'ml' / 'value' / 'campaign-c3'
TEAM = 'Rain-Recife-special-stat-fix'
CANDIDATE = ROOT / 'data' / 'ml' / 'human' / 'models-v4' / 'human-cf8949974b6ea497.json'
W, TEMP = 0.2, 1.0
STAGES = {'dev': 400, 'confirm': 400, 'standard': 200}


def rows(stage: str) -> list[dict]:
    p = CAMP / 'gates' / f'c3-{stage}.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def status() -> dict:
    out = {}
    for stage, pairs in STAGES.items():
        r = rows(stage)
        s = summarize(r, pairs, 0.05) if r else None
        if s:
            s['cand_wins'] = sum(1 for x in r if x['arm'] == 'cand' and x.get('win'))
            s['inc_wins'] = sum(1 for x in r if x['arm'] == 'inc' and x.get('win'))
            s['games'] = sum(1 for x in r if x.get('score') is not None)
            s['seconds_per_game'] = round(sum(x.get('seconds', 0) for x in r) / max(1, len(r)))
        out[stage] = s
    std = out.get('standard')
    harm = False
    if std and std['complete']:
        g, l = std['gains'], std['losses']
        p_harm = sum(comb(g + l, i) for i in range(l, g + l + 1)) / 2 ** (g + l) if g + l else 1.0
        harm = l > g and p_harm <= 0.05
        std['p_harm'] = round(p_harm, 5)
    out['promotable'] = bool(out.get('dev') and out['dev']['complete'] and out['dev']['passed']
                             and out.get('confirm') and out['confirm']['complete'] and out['confirm']['passed']
                             and not harm)
    out['standard_blocks'] = harm
    return out


def promote(force: bool = False) -> dict:
    from search.human.store import opp_pointer, promote_opp, weights_sha
    from search.mlops import _log_campaign
    st = status()
    if not st['promotable'] and not force:
        raise SystemExit(json.dumps({'refused': 'protocol condition not met', 'status': st}))
    sha = weights_sha(CANDIDATE)
    if not sha.startswith('cf8949974b6ea497'):
        raise SystemExit('candidate file changed: ' + sha)
    evidence = {'campaign': 'c3-20261013', 'protocol': str(CAMP / 'PROTOCOL.md'),
                'dev': st['dev'], 'confirm': st['confirm'], 'standard': st.get('standard'), 'forced': force}
    rec = promote_opp(CANDIDATE, W, TEMP, evidence, opp_pointer(TEAM))
    _log_campaign({'event': 'promote', 'id': 'c3-20261013', 'at': time.time(), 'host': 'DiveMac', 'team': TEAM,
                   'kind': 'opp_model', 'pointer': str(opp_pointer(TEAM)), 'record': rec})
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['status', 'promote'])
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'status':
        print(json.dumps(status(), indent=1))
    else:
        print(json.dumps(promote(args.force), indent=1))


if __name__ == '__main__':
    main()
