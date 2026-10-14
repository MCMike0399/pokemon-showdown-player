"""Human decisions from public battle logs -> training rows for an opponent policy model.

    .venv/bin/python search/human/extract.py --replays 'data/replays/regmc/replays-*.jsonl' \
        --out data/ml/human/regmc [--workers 6] [--ledgers]

A public log shows what each player did every turn. For each move turn and each
player ("actor"), the position is rebuilt the way the search rebuilds it
(search/tracker.py state up to that turn; search/priors.py samples the hidden
sets), with one difference: the actor knows their own team, so their revealed
moves, items, abilities and Mega stone from the WHOLE game are used for their
own side, while the other side is sampled from what the actor had seen so far.
The engine then enumerates the actor's legal joint actions (search/engine.cjs
`human`) with the features of search/screenfeat.cjs, and the action the actor
really took is the label.

Output shards (npz): S (R, F) position features from the actor's side, A (N, D)
action features, rec (N,) row of S, y (N,) 1 for the taken action, and per
decision rating, turn, game id, actor side, hand evaluation and the choice
strings. Rows are grouped per decision so a listwise (softmax) loss applies.
Also writes previews.jsonl: both sixes, the leads and brought Pokemon, result
and ratings, for a team-preview model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from battle_state import to_id  # noqa: E402
from search.priors import sample_bench, sample_set  # noqa: E402
from search.tracker import PROTECT_MOVES, Tracker  # noqa: E402

FMT = 'gen9championsvgc2026regmc'
CALLED = ('[from]',)


def _base(species: str) -> str:
    return re.sub(r'-Mega(-[XYZ])?$', '', species.split(',')[0].strip())


def _volatiles(mon) -> dict:
    return {k: v for k, v in (mon.volatiles or {}).items() if v is not None} if mon else {}


# ------------------------------------------------------------------ log parsing
def split_turns(log: list[str]) -> list[tuple[int, int, int]]:
    """(turn, start index of the |turn| line, end index exclusive) for every turn."""
    out = []
    starts = [(i, int(l.split('|')[2])) for i, l in enumerate(log) if l.startswith('|turn|')]
    for k, (i, t) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(log)
        out.append((t, i, end))
    return out


def turn_actions(lines: list[str]) -> dict[str, dict]:
    """Voluntary actions of the turn by slot ('p1a'...): {kind, move, target, mega, species, unknown}.

Lines run from after the |turn| line to the next |turn|. Only what happens before
|upkeep| is a chosen action; later switches replace fainted Pokemon."""
    acts: dict[str, dict] = {}
    moved: set[str] = set()
    pending_mega: set[str] = set()
    for line in lines:
        p = line.split('|')
        if len(p) < 2:
            continue
        kind = p[1]
        if kind == 'upkeep' or kind in ('win', 'tie'):
            break
        if kind == '-mega' and len(p) > 2:
            pending_mega.add(p[2].split(':')[0].strip())
        elif kind == 'move' and len(p) > 3:
            slot = p[2].split(':')[0].strip()
            rest = '|'.join(p[4:])
            if slot in moved:
                continue  # Dancer / Instruct / second action: not a choice
            moved.add(slot)
            if '[from]' in rest and 'lockedmove' not in rest:
                acts[slot] = {'kind': 'unknown'}  # called by another move/item
                continue
            move = to_id(p[3])
            if move == 'struggle' or 'lockedmove' in rest:
                acts[slot] = {'kind': 'unknown'}
                continue
            target = p[4].strip() if len(p) > 4 and p[4] and not p[4].startswith('[') else ''
            acts[slot] = {'kind': 'move', 'move': move, 'target': target.split(':')[0].strip() if target else '',
                          'mega': slot in pending_mega, 'notarget': '[notarget]' in rest or '[still]' in rest}
        elif kind == 'cant' and len(p) > 2:
            slot = p[2].split(':')[0].strip()
            if slot not in moved:
                moved.add(slot)
                acts[slot] = {'kind': 'unknown'}
        elif kind in ('switch', 'drag') and len(p) > 3:
            slot = p[2].split(':')[0].strip()
            if kind == 'drag' or slot in moved:
                moved.add(slot)
                if slot not in acts:
                    acts[slot] = {'kind': 'unknown'}
                continue
            moved.add(slot)
            acts[slot] = {'kind': 'switch', 'species': p[3].split(',')[0].strip()}
    return acts


def player_info(log: list[str]) -> dict:
    out = {'p1': {}, 'p2': {}}
    for line in log:
        if line.startswith('|player|'):
            p = line.split('|')
            if len(p) > 3 and p[2] in out and p[3]:
                out[p[2]]['name'] = p[3]
                if len(p) > 5 and p[5].isdigit():
                    out[p[2]]['rating'] = int(p[5])
        elif line.startswith('|win|'):
            out['winner'] = line.split('|')[2]
        elif line.startswith('|tie'):
            out['winner'] = None
    return out


# ------------------------------------------------------------------ world building
def _mon_key(m):
    return (m.side, m.name)


def side_world(state: Tracker, know: Tracker, side: str, rng: random.Random, fmt: str = FMT, cutoff: str = '0') -> list[dict]:
    """One side of a determinized world: state from `state` (what was visible at the
turn), hidden sets conditioned on `know` (the whole game for the actor's own side,
the visible prefix for the other side), the remainder sampled from priors."""
    actives = [state.slots.get(side + 'a'), state.slots.get(side + 'b')]
    ordered = []
    for m in actives:
        if m is not None and m not in ordered:
            ordered.append(m)
    for m in state.side_mons(side):
        if m not in ordered:
            ordered.append(m)
    # Pokemon the actor brought that have not appeared yet (known from the full game).
    seen = {m.name for m in ordered}
    later = [m for m in know.side_mons(side) if m.name not in seen] if know is not state else []
    preview = state.preview.get(side) or []
    known_names = {_base(m.species) for m in ordered} | {_base(m.species) for m in later}
    need = max(0, 4 - len(ordered) - len(later))
    extra = sample_bench(rng, preview, list(known_names), need, fmt, cutoff) if preview and need else []
    mega_used = state.mega_used.get(side)
    sheet = {to_id(_base(x.get('species') or x.get('name') or '')): x for x in (know.sheets.get(side) or [])}

    def from_sheet(species, set_):
        x = sheet.get(to_id(_base(species)))
        if not x:
            return set_
        out = dict(set_)
        out['moves'] = [to_id(mv) for mv in x.get('moves') or []] or set_['moves']
        if x.get('item'):
            out['item'] = to_id(x['item'])
        if x.get('ability'):
            out['ability'] = to_id(x['ability'])
        if x.get('nature'):
            out['nature'] = x['nature']
        return out

    out = []
    for m in ordered[:4]:
        k = know.mons.get(_mon_key(m), m)  # full-game knowledge of this Pokemon's set
        observed = {'moves': k.moves, 'item': None if not k.item_known else k.item, 'ability': k.ability,
                    'mega': m.mega, 'mega_species': m.species if m.mega else None,
                    'mega_ruled_out': mega_used and not m.mega}
        set_ = sample_set(rng, m.base_species, observed, fmt, cutoff)
        if k.item_known and k.item and not m.mega:
            set_['item'] = k.item
        set_ = from_sheet(m.base_species, set_)
        st = {
            'hp': m.hp, 'fainted': m.fainted, 'status': m.status,
            'boosts': dict(m.boosts) if m.active_slot else {},
            'volatiles': _volatiles(m) if m.active_slot else {},
            'item': '' if m.item_gone else None,
            'item_gone': m.item_gone,
            'ability': m.ability if not m.mega else None,
            'mega': m.mega,
            'actions': m.actions if m.active_slot else 0,
            'active_turns': (state.turn - m.switch_turn) if m.active_slot else 0,
            'protect_last_turn': bool(m.active_slot and m.last_move in PROTECT_MOVES and
                                      m.last_move_turn == state.turn - 1),
            'sleep_turns': m.sleep_turns, 'toxic_turns': m.toxic_turns, 'turn': state.turn,
            'locked_move': m.last_move if (m.active_slot and to_id(set_.get('item', '')).startswith('choice')
                                           and m.last_move and not m.item_gone) else None,
        }
        out.append({'set': set_, 'state': st})
    for m in later[:max(0, 4 - len(out))]:
        observed = {'moves': m.moves, 'item': None if not m.item_known else m.item, 'ability': m.ability,
                    'mega_ruled_out': mega_used}
        set_ = from_sheet(m.base_species, sample_set(rng, m.base_species, observed, fmt, cutoff))
        if m.item_known and m.item and not m.mega:
            set_['item'] = m.item
        out.append({'set': set_, 'state': {'hp': 1.0, 'turn': state.turn}})
    for species in extra:
        set_ = from_sheet(species, sample_set(rng, species, {'mega_ruled_out': mega_used}, fmt, cutoff))
        out.append({'set': set_, 'state': {'hp': 1.0, 'turn': state.turn}})
    return out


def field_world(state: Tracker, me: str) -> dict:
    field = state.field_state()
    foe = 'p2' if me == 'p1' else 'p1'
    return {'turn': max(1, state.turn), 'weather': field['weather'], 'terrain': field['terrain'],
            'pseudo': field['pseudo'], 'sides': {'p1': field['sides'].get(me, {}), 'p2': field['sides'].get(foe, {})},
            'mega_used': {'p1': state.mega_used.get(me), 'p2': state.mega_used.get(foe)}}


# ------------------------------------------------------------------ labelling
def match_choice(out: dict, acts: dict, me: str, party_names: list[str], alive: list[bool]) -> tuple[list[int], str]:
    """Indices of the engine choices consistent with what the log shows the actor did.

One index when both slots' actions are known. A slot whose Pokemon fainted or
flinched before acting (no logged action, or an unknown one) is marginalised: any
move part is consistent (switches resolve before moves, so it was not a switch),
restricted to Mega parts when the log shows it Mega Evolved."""
    parts = []
    for s, letter in enumerate('ab'):
        slot = me + letter
        act = acts.get(slot)
        moves = out['moves'][s] if s < len(out['moves']) else []
        if act is None and not (s < len(alive) and alive[s]):
            parts.append(('exact', ['pass']))
            continue
        if act is None or act['kind'] == 'unknown':
            parts.append(('any_move', [' mega'] if (act or {}).get('mega') else None))
            continue
        if act['kind'] == 'switch':
            idx = [i for i, n in enumerate(party_names) if _base(n) == _base(act['species'])]
            if not idx:
                return [], 'switch_target_missing'
            parts.append(('exact', [f'switch {idx[0] + 1}']))
            continue
        mid = act['move']
        if mid not in moves:
            return [], 'move_missing'
        j = moves.index(mid) + 1
        t = act['target']
        tnum = None
        if t:
            if t[:2] == me:
                tnum = -1 if t.endswith('a') else -2
            else:
                tnum = 1 if t.endswith('a') else 2
        mega = ' mega' if act.get('mega') else ''
        cands = []
        if tnum is not None:
            cands.append(f'move {j} {tnum}{mega}')
        cands.append(f'move {j}{mega}')
        if act.get('notarget') or tnum is None:
            cands += [f'move {j} 1{mega}', f'move {j} 2{mega}']
        parts.append(('exact', cands))
    choices = out['choices']
    split = [[x.strip() for x in c.split(',')] for c in choices]

    def part_ok(kind, spec, text):
        if kind == 'exact':
            return text in spec
        return text.startswith('move') and (spec is None or text.endswith(spec[0]))

    # The first consistent exact candidate per slot wins (a targeted form before an untargeted one).
    for s, (kind, spec) in enumerate(parts):
        if kind == 'exact' and len(spec) > 1:
            present = [c for c in spec if any(len(sp) > s and sp[s] == c for sp in split)]
            if present:
                parts[s] = ('exact', [present[0]])
    ok = [i for i, sp in enumerate(split)
          if len(sp) == len(parts) and all(part_ok(kind, spec, sp[s]) for s, (kind, spec) in enumerate(parts))]
    if not ok:
        return [], 'not_enumerated'
    return ok, 'ok' if all(k == 'exact' for k, _ in parts) else 'marginal'


# ------------------------------------------------------------------ driver
class Worker:
    def __init__(self, seed: int = 0, fmt: str = FMT):
        from search.agent import Engine
        self.engine = Engine()
        self.fmt = fmt
        self.rng = random.Random(seed)

    async def game(self, gid: str, log: list[str], min_turn: int = 1, exclude_name: str | None = None) -> tuple[list[dict], dict, dict]:
        info = player_info(log)
        turns = split_turns(log)
        full = {s: Tracker(s).feed(log) for s in ('p1', 'p2')}
        rows, stats = [], {}
        for turn, i, end in turns:
            if turn < min_turn:
                continue
            prefix = log[:i + 1]
            acts = turn_actions(log[i + 1:end])
            trackers = {s: Tracker(s).feed(prefix) for s in ('p1', 'p2')}
            for me in ('p1', 'p2'):
                if exclude_name and info[me].get('name') == exclude_name:
                    continue
                st = trackers[me]
                alive = [bool((m := st.slots.get(me + x)) and not m.fainted) for x in 'ab']
                if not any(k.startswith(me) for k in acts):
                    stats['no_action'] = stats.get('no_action', 0) + 1
                    continue
                if all(acts.get(me + x, {}).get('kind', 'unknown') == 'unknown' for x in 'ab' if alive['ab'.index(x)]):
                    stats['unknown'] = stats.get('unknown', 0) + 1
                    continue
                foe = 'p2' if me == 'p1' else 'p1'
                try:
                    world = {'p1': side_world(st, full[me], me, self.rng, self.fmt),
                             'p2': side_world(st, st, foe, self.rng, self.fmt)}
                    field = field_world(st, me)
                    out = await self.engine.call({'type': 'human', 'format': self.fmt, 'field': field, 'worlds': [world]}, timeout=60)
                except Exception as error:
                    stats['rebuild_error'] = stats.get('rebuild_error', 0) + 1
                    stats.setdefault('errors', []).append(repr(error)[:160])
                    continue
                if 'error' in out or not out.get('choices'):
                    stats['engine_error'] = stats.get('engine_error', 0) + 1
                    continue
                mask, why = match_choice(out, acts, me, out['party'], alive)
                stats[why] = stats.get(why, 0) + 1
                if not mask:
                    continue
                if len(out['choices']) < 2:
                    stats['single_choice'] = stats.get('single_choice', 0) + 1
                    continue
                rows.append({'game': gid, 'side': me, 'turn': turn, 'rating': info[me].get('rating'),
                             'win': None if info.get('winner') is None else int(info['winner'] == info[me].get('name')),
                             'sx': out['sx'], 'a': out['a'], 'mask': mask, 'exact': why == 'ok',
                             'choices': out['choices'], 'hand': out['hand'], 'n': len(out['choices'])})
        return rows, stats, info

    async def close(self):
        await self.engine.close()


def preview_record(gid: str, log: list[str], info: dict) -> dict | None:
    six = {'p1': [], 'p2': []}
    brought = {'p1': [], 'p2': []}
    leads = {'p1': [], 'p2': []}
    started = False
    for line in log:
        p = line.split('|')
        if len(p) < 2:
            continue
        if p[1] == 'start':
            started = True
            continue
        if len(p) < 3:
            continue
        if p[1] == 'poke' and p[2] in six:
            six[p[2]].append(p[3].split(',')[0].strip())
        elif p[1] in ('switch', 'drag', 'replace') and started and len(p) > 3:
            side = p[2][:2]
            sp = _base(p[3].split(',')[0].strip())
            if sp not in brought[side]:
                brought[side].append(sp)
            if len(leads[side]) < 2 and p[1] == 'switch' and len(brought[side]) <= 2:
                leads[side].append(sp)
    if len(six['p1']) != 6 or len(six['p2']) != 6:
        return None
    return {'game': gid, 'six': six, 'brought': brought, 'leads': leads, 'winner': info.get('winner'),
            'names': {s: info[s].get('name') for s in ('p1', 'p2')},
            'rating': {s: info[s].get('rating') for s in ('p1', 'p2')}}


def write_shard(rows: list[dict], path: Path):
    S, A, rec, y, meta = [], [], [], [], []
    for r in rows:
        k = len(S)
        S.append(r['sx'])
        ok = set(r['mask'])
        for i, a in enumerate(r['a']):
            A.append(a); rec.append(k); y.append(1 if i in ok else 0)
        meta.append({'game': r['game'], 'side': r['side'], 'turn': r['turn'], 'rating': r['rating'], 'win': r['win'],
                     'hand': r['hand'], 'n': r['n'], 'exact': r['exact'], 'mask': r['mask'], 'choices': r['choices']})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(tmp, S=np.asarray(S, np.float32), A=np.asarray(A, np.float32), rec=np.asarray(rec, np.int32),
                        y=np.asarray(y, np.int8), meta=np.asarray(json.dumps(meta)))
    os.replace(tmp, path)


def iter_replays(patterns: list[str]):
    for pattern in patterns:
        for f in sorted(glob.glob(pattern)):
            for line in open(f):
                try:
                    rep = json.loads(line)
                except ValueError:
                    continue
                log = rep.get('log') or ''
                if isinstance(log, str):
                    log = log.split('\n')
                yield rep['id'], log, {'rating': rep.get('rating'), 'uploadtime': rep.get('uploadtime')}


def iter_ledgers(runs: Path):
    """Our own live games: the terminal event keeps the full public log (both sides act)."""
    import ast
    for path in sorted(runs.glob('*/events.jsonl')):
        for line in path.open():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('kind') != 'terminal':
                continue
            log = e.get('log')
            if isinstance(log, str):
                try:
                    log = ast.literal_eval(log)
                except (ValueError, SyntaxError):
                    continue
            room = (e.get('result') or {}).get('room')
            if room and isinstance(log, list):
                yield 'live-' + room, log, {'time': e.get('time')}


def _run_shard(args):
    import asyncio
    items, out_path, seed, min_turn, exclude = args
    if Path(out_path).exists():
        return {'skipped': str(out_path)}
    async def go():
        w = Worker(seed)
        rows, previews, stats = [], [], {}
        try:
            for gid, log, extra in items:
                r, s, info = await w.game(gid, log, min_turn, exclude)
                rows.extend(r)
                for k, v in s.items():
                    if k != 'errors':
                        stats[k] = stats.get(k, 0) + v
                    else:
                        stats.setdefault('errors', []).extend(v[:2])
                pv = preview_record(gid, log, info)
                if pv:
                    pv['replay'] = extra
                    previews.append(pv)
        finally:
            await w.close()
        if rows:
            write_shard(rows, Path(out_path))
        with open(str(out_path).replace('.npz', '.previews.jsonl'), 'w') as fh:
            for pv in previews:
                fh.write(json.dumps(pv) + '\n')
        stats['rows'] = len(rows)
        stats['games'] = len(items)
        stats['errors'] = stats.get('errors', [])[:3]
        return stats
    return asyncio.run(go())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replays', action='append', default=[], help='glob of replay jsonl shards')
    ap.add_argument('--ledgers', action='store_true', help='also our own live games (data/ml/browser-runs)')
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'ml' / 'human' / 'regmc')
    ap.add_argument('--games-per-shard', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--min-turn', type=int, default=1)
    ap.add_argument('--shard-size', type=int, default=1000, help='lines of a complete replay file')
    ap.add_argument('--include-partial', action='store_true')
    ap.add_argument('--exclude-name', default=None, help="never treat this player as a human actor (our account; default from .env PS_USERNAME)")
    args = ap.parse_args()
    if args.exclude_name is None:
        for line in (ROOT / '.env').read_text().splitlines() if (ROOT / '.env').exists() else []:
            if line.startswith('PS_USERNAME='):
                args.exclude_name = line.split('=', 1)[1].strip().strip('"\'')
    from multiprocessing import Pool
    jobs = []
    # One output shard per source file (and chunk), so a rerun with more replay files
    # only processes the new ones. A replay file still being appended is skipped
    # unless --include-partial, because its shard would otherwise be final.
    files = []
    for pattern in args.replays:
        files.extend(sorted(glob.glob(pattern)))
    for f in files:
        name = Path(f).stem
        rows = [l for l in open(f)]
        if len(rows) < args.shard_size and not args.include_partial:
            continue
        items = list(iter_replays([f]))
        if args.limit:
            items = items[:args.limit]
        for c in range(0, len(items), args.games_per_shard):
            jobs.append((items[c:c + args.games_per_shard], args.out / f'replay-{name}-{c // args.games_per_shard:02d}.npz',
                         c, args.min_turn, args.exclude_name))
    if args.ledgers:
        items = list(iter_ledgers(ROOT / 'data' / 'ml' / 'browser-runs'))
        for c in range(0, len(items), args.games_per_shard):
            jobs.append((items[c:c + args.games_per_shard], args.out / f'live-{c // args.games_per_shard:03d}.npz',
                         c, args.min_turn, args.exclude_name))
    jobs = [j for j in jobs if not Path(j[1]).exists()]
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total = {}
    with Pool(args.workers) as pool:
        for stats in pool.imap_unordered(_run_shard, jobs):
            for k, v in stats.items():
                if isinstance(v, int):
                    total[k] = total.get(k, 0) + v
            print(json.dumps({'shard': stats, 'elapsed': round(time.time() - t0)}), flush=True)
    print(json.dumps({'total': total, 'seconds': round(time.time() - t0)}))


if __name__ == '__main__':
    main()
