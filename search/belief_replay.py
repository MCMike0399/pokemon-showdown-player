"""Replay frozen public decision contexts through the baseline and belief agent.

Input: JSON list of {ctx, ...metadata}; ctx holds request, legal choices, own
team_sets and the public_log PREFIX available at the decision. Never use future
events in that prefix. This is a legality/latency check, not a win-rate test.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from search.agent import SearchAgent  # noqa: E402
from search.belief_audit import OPTIONS, quantiles  # noqa: E402
from search.config import agent_kwargs  # noqa: E402


async def replay(source, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    protocol = {'input_sha256': hashlib.sha256(raw).hexdigest(), 'seed': 20261024,
                'options': OPTIONS, 'base': agent_kwargs(), 'source_hashes': {
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                    ('search/agent.py', 'search/engine.cjs', 'search/belief.cjs')}}
    manifest = out / 'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('source or inputs changed; use a new output directory')
    manifest.write_text(json.dumps(protocol, indent=2))
    (out / 'inputs.json').write_bytes(raw)
    rows = []
    for i, item in enumerate(json.loads(raw)):
        dest = out / f'{i:03}.json'
        if dest.exists():
            rows.append(json.loads(dest.read_text()))
            continue
        row = {'index': i, 'metadata': {k: v for k, v in item.items() if k != 'ctx'}}
        for arm in (('inc', 'cand') if i % 2 == 0 else ('cand', 'inc')):
            agent = SearchAgent(seed=20261024 + i, **agent_kwargs(
                **({'root_policy': OPTIONS} if arm == 'cand' else {})))
            try:
                choice = await agent.decide(item['ctx'])
                row[arm] = {'choice': choice, 'legal': choice in item['ctx']['choices'],
                            'stats': agent.stats, 'trace': agent.last_trace}
            finally:
                await agent.close()
        dest.write_text(json.dumps(row))
        rows.append(row)
        print(json.dumps({'index': i, 'changed': row['inc']['choice'] != row['cand']['choice'],
                          'candidate_ms': row['cand']['stats']['ms'],
                          'errors': row['cand']['stats']['errors']}), flush=True)
    summary = {'n': len(rows), 'scope': 'Live-context replay, not observed wins or Elo',
               'arms': {a: {'latency_ms': quantiles([r[a]['stats']['ms'] for r in rows]),
                            'fallbacks': sum(r[a]['stats']['fallbacks'] for r in rows),
                            'root_fallbacks': sum(r[a]['stats'].get('root_fallbacks', 0) for r in rows),
                            'illegal': sum(not r[a]['legal'] for r in rows)} for a in ('inc', 'cand')},
               'changed': sum(r['inc']['choice'] != r['cand']['choice'] for r in rows)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    asyncio.run(replay(args.source, args.out))


if __name__ == '__main__':
    main()
