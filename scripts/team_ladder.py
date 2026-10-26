#!/usr/bin/env python3
"""Campaign c13: a king-of-the-hill team ladder on the live account (docs/c13-team-ladder.md).

Every round is one live interleaved A/B (search/ab_test.py) with a single factor `team`:
the base arm plays the champion team, the other arm plays one challenger, and every game
draws its arm, so both teams meet the same opponents at the same rating. The round ends
when the factor's SPRT decides (or at max_games). H1 makes the challenger the champion.
The next challenger then faces whoever is champion. After the last round a new champion
keeps playing through a final spec with nothing randomised; with no new champion the spec
is set inactive and the runner plays its own --team again.

The search configuration never changes here. Teams are `teams.json` names. The controller
only writes the A/B spec and its own state; it refuses to replace an active spec that
belongs to another experiment.

    .venv/bin/python scripts/team_ladder.py init --dir data/ml/value/campaign-c13 \\
        --champion Rain-Recife-special-stat-fix --challengers A B
    .venv/bin/python scripts/team_ladder.py run --dir data/ml/value/campaign-c13
    .venv/bin/python scripts/team_ladder.py status --dir data/ml/value/campaign-c13
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from search import ab_test  # noqa: E402

PREFIX = 'c13'
SPRT = {'elo0': 0.0, 'elo1': 40.0, 'alpha': 0.05, 'beta': 0.10, 'min_games': 300, 'max_games': 3000}


def now() -> str:
    return datetime.now().isoformat(timespec='seconds')


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def log(out: Path, message: str) -> None:
    with (out / 'ladder.log').open('a') as fh:
        fh.write(f'{now()} {message}\n')


def round_spec(state: dict) -> dict:
    n = state['round']
    return {'id': f'{PREFIX}r{n}', 'active': True, 'seed': state['seed'] + n, 'created': now(),
            'protocol': state['protocol'], **SPRT, 'base': {'team': state['champion']},
            'factors': {'team': {'team': state['challenger']}}, 'frozen': {}, 'history': []}


def final_spec(state: dict) -> dict:
    # Nothing randomised: every game plays the champion (validate() needs one factor).
    return {'id': f'{PREFIX}final', 'active': True, 'seed': state['seed'], 'created': now(),
            'protocol': state['protocol'], **SPRT, 'base': {'team': state['champion']},
            'factors': {'team': {'team': state['champion']}}, 'frozen': {'team': True}, 'history': []}


def ours(raw: dict | None) -> bool:
    return bool(raw) and str(raw.get('id', '')).startswith(PREFIX)


def init(out: Path, spec_path: Path, champion: str, challengers: list[str], teams: dict, seed: int,
         protocol: str) -> dict:
    for name in [champion, *challengers]:
        if name not in teams:
            raise ValueError(f'unknown team {name!r} (teams.json)')
    if len(set([champion, *challengers])) != len(challengers) + 1:
        raise ValueError('teams must be distinct')
    try:
        current = json.loads(spec_path.read_text())
    except (OSError, ValueError):
        current = None
    if current and current.get('active') and not ours(current):
        raise ValueError(f"spec {current.get('id')!r} is active; finish that experiment first")
    out.mkdir(parents=True, exist_ok=True)
    state = {'phase': 'running', 'created': now(), 'seed': seed, 'protocol': protocol,
             'original': champion, 'champion': champion, 'challenger': challengers[0],
             'queue': list(challengers[1:]), 'round': 1, 'history': []}
    write_json(out / 'state.json', state)
    spec = round_spec(state)
    ab_test.validate(spec)
    ab_test.write_spec(spec, spec_path)
    log(out, f"init: champion {champion}, round 1 challenger {challengers[0]}, queue {state['queue']}")
    return state


def step(out: Path, spec_path: Path, runs_dir: Path | None = None) -> str:
    """One poll: 'running', 'next' (a round was decided), 'done' or 'stopped'."""
    state = json.loads((out / 'state.json').read_text())
    if state['phase'] != 'running':
        return 'done'
    raw = json.loads(spec_path.read_text())
    want = f"{PREFIX}r{state['round']}"
    if raw.get('id') != want:
        raise RuntimeError(f"spec is {raw.get('id')!r}, expected {want!r}; not touching it")
    if not raw.get('active'):
        log(out, f'{want}: spec set inactive by hand; campaign stopped')
        state.update(phase='stopped', stopped=now())
        write_json(out / 'state.json', state)
        return 'stopped'
    st = ab_test.status(spec_path, runs_dir)
    s = st['factors']['team']
    write_json(out / 'status.json', {'time': now(), 'round': state['round'], 'champion': state['champion'],
                                     'challenger': state['challenger'], 'queue': state['queue'], **s})
    if not s['decision']:
        return 'running'
    won = s['decision'] == 'H1'
    state['history'].append({'round': state['round'], 'time': now(), 'champion': state['champion'],
                             'challenger': state['challenger'], 'decision': s['decision'], 'stats': s})
    raw.update(active=False, frozen={'team': won}, finished=now())
    ab_test.write_spec(raw, spec_path)
    log(out, f"{want}: {s['decision']} after {s['games']} games, {state['challenger']} vs {state['champion']}: "
             f"Elo {s['elo']} (score {s['score_on']} vs {s['score_off']}), LLR {s['llr']}")
    if won:
        state['champion'] = state['challenger']
    if state['queue']:
        state.update(challenger=state['queue'].pop(0), round=state['round'] + 1)
        write_json(out / 'state.json', state)
        ab_test.write_spec(round_spec(state), spec_path)
        log(out, f"round {state['round']}: {state['challenger']} challenges {state['champion']}")
        return 'next'
    state.update(phase='done', finished=now(), challenger=None)
    write_json(out / 'state.json', state)
    if state['champion'] != state['original']:
        ab_test.write_spec(final_spec(state), spec_path)
        log(out, f"done: champion {state['champion']} plays every game (spec {PREFIX}final); "
                 f"make it the runner's --team to keep it without the spec")
    else:
        log(out, f"done: {state['original']} stays; spec inactive")
    return 'done'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init', 'run', 'status'])
    ap.add_argument('--dir', type=Path, required=True)
    ap.add_argument('--spec', type=Path, default=ab_test.SPEC)
    ap.add_argument('--teams-file', type=Path, default=ROOT / 'teams.json')
    ap.add_argument('--champion')
    ap.add_argument('--challengers', nargs='+')
    ap.add_argument('--seed', type=int, default=20261300)
    ap.add_argument('--protocol', default='docs/c13-team-ladder.md')
    ap.add_argument('--poll', type=float, default=1800)
    args = ap.parse_args(argv)
    out = (ROOT / args.dir).resolve()
    if args.command == 'init':
        if not args.champion or not args.challengers:
            ap.error('init needs --champion and --challengers')
        init(out, args.spec, args.champion, args.challengers, json.loads(args.teams_file.read_text()),
             args.seed, args.protocol)
        return 0
    if args.command == 'status':
        state = json.loads((out / 'state.json').read_text())
        print(json.dumps({k: state[k] for k in ('phase', 'round', 'champion', 'challenger', 'queue')}, indent=1))
        for h in state['history']:
            s = h['stats']
            print(f"round {h['round']}: {h['challenger']} vs {h['champion']}: {h['decision']} after {s['games']} "
                  f"games, Elo {s['elo']}, LLR {s['llr']}")
        if state['phase'] == 'running':
            print(ab_test.format_status(ab_test.status(args.spec)))
        return 0
    log(out, 'controller started')
    while True:
        try:
            result = step(out, args.spec)
        except Exception as error:  # noqa: BLE001 - a ledger or the spec can be mid-write
            log(out, f'check failed, retrying: {error!r}'[:500])
            result = 'running'
        if result in ('done', 'stopped'):
            return 0
        time.sleep(5 if result == 'next' else args.poll)


if __name__ == '__main__':
    sys.exit(main())
