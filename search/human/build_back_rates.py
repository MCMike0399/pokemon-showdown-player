"""Corpus bring rates for the opponent's unseen back pair (search/priors.py `sample_bench`).

    .venv/bin/python search/human/build_back_rates.py [--replays 'data/replays/*/replays-*.jsonl'] \
        [--out search/back_rates.json] [--exclude-file NAMES.txt]

With the leads known, which two of the other four did the opponent bring? The
deployed prior weights candidates by ladder usage, so Rillaboom (1.41M) outweighed
Kommo-o (80k) about 18 to 1, while every species is brought at a similar rate
(0.47-0.71). On October replays that prior scored log loss 2.47 for the back pair,
worse than uniform (1.79); per-species back rates plus lead pairing score 1.65.

Counts, per species `s` on a six and not leading: `back[s]` games it was brought,
`nonlead[s]` games in total; per (s, lead) the same counts, for a lead-pair
adjustment. Our own account is excluded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from battle_state import to_id  # noqa: E402
from search.human.extract import iter_replays, player_info, preview_record  # noqa: E402

OUT = ROOT / 'search' / 'back_rates.json'


def _sid(name: str) -> str:
    return to_id(re.sub(r'-Mega(-[XYZ])?$', '', name.split(',')[0].strip()))


def build(patterns: list[str], exclude: set[str]) -> dict:
    back, nonlead = Counter(), Counter()
    pair_back, pair_on = Counter(), Counter()
    seen, games = set(), 0
    for gid, log, _ in iter_replays(patterns):
        if gid in seen:
            continue
        seen.add(gid)
        info = player_info(log)
        pv = preview_record(gid, log, info)
        if not pv:
            continue
        games += 1
        for side in ('p1', 'p2'):
            if to_id(pv['names'].get(side) or '') in exclude:
                continue
            brought = {_sid(x) for x in pv['brought'][side]}
            leads = [_sid(x) for x in pv['leads'][side]]
            # Only games long enough to show all four (else "not brought" is unknown).
            if len(brought) != 4 or len(leads) != 2:
                continue
            for s in {_sid(x) for x in pv['six'][side]} - set(leads):
                hit = int(s in brought)
                nonlead[s] += 1
                back[s] += hit
                for lead in leads:
                    pair_on[s + '>' + lead] += 1
                    pair_back[s + '>' + lead] += hit
    return {'version': 1, 'games': games, 'back': dict(back), 'nonlead': dict(nonlead),
            'pair': {k: [pair_back[k], n] for k, n in pair_on.items() if n >= 20}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replays', action='append', default=[])
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--exclude-file', type=Path, help='one player name per line (likely bot accounts)')
    args = ap.parse_args()
    patterns = args.replays or [str(ROOT / 'data' / 'replays' / '*' / 'replays-*.jsonl')]
    exclude = set()
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('PS_USERNAME='):
                exclude.add(to_id(line.split('=', 1)[1].strip().strip('"\'')))
    if args.exclude_file:
        exclude |= {to_id(x) for x in args.exclude_file.read_text().split() if x.strip()}
    table = build(patterns, exclude)
    args.out.write_text(json.dumps(table, separators=(',', ':'), sort_keys=True))
    print(json.dumps({'games': table['games'], 'species': len(table['back']), 'pairs': len(table['pair']),
                      'out': str(args.out)}))


if __name__ == '__main__':
    main()
