"""Real human teams: live opponents' open team sheets + tournament research teams.
Sheets omit stat points; draw them from ladder usage spreads matching the nature."""
import json, random, sqlite3, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from battle_state import unpack_team, to_id
from search.build_ladder_teams import our_side
from search.priors import load_usage, _base_id, STAT_ORDER
FMT = 'gen9championsvgc2026regmc'

def spread_for(rng, species, nature):
    u = load_usage(FMT, '0'); entries = u['by_base'].get(_base_id(species, u['dex']), [])
    pool = [(k, w) for e in entries for k, w in e['spreads'] if not nature or k.split(':')[0].lower() == (nature or '').lower()]
    if not pool:
        pool = [(k, w) for e in entries for k, w in e['spreads']]
    if not pool:
        return {'hp': 32, 'atk': 2, 'spa': 32}
    tot = sum(w for _, w in pool); r = rng.random() * tot; acc = 0
    for k, w in pool:
        acc += w
        if r <= acc:
            vals = [int(x) for x in k.split(':')[1].split('/')]
            return dict(zip(STAT_ORDER, vals))
    return dict(zip(STAT_ORDER, [int(x) for x in pool[-1][0].split(':')[1].split('/')]))

def main(out):
    rng = random.Random(11); teams = []; seen = set()
    db = sqlite3.connect(f'file:{ROOT}/data/ml/experience.sqlite3?mode=ro', uri=True)
    for (log,) in db.execute("select log from public_battles where source='own-live-game'"):
        log = json.loads(log); me = our_side(log)
        for l in log:
            if l.startswith('|showteam|') and l.split('|')[2] != me:
                t = unpack_team(l.split('|', 3)[3])
                key = tuple(sorted(to_id(m['species']) for m in t) + sorted(to_id(m['item']) for m in t))
                if key in seen: continue
                seen.add(key)
                sets = [{'species': m['species'], 'item': m['item'], 'ability': m['ability'], 'moves': m['moves'],
                         'nature': m['nature'] or 'Serious', 'level': 50, 'evs': spread_for(rng, m['species'], m['nature'])} for m in t]
                teams.append({'source': 'live-ots', 'sets': sets})
    for player, event, sets in db.execute("select player, event, sets from research_teams where format=?", (FMT,)):
        sets = json.loads(sets)
        for s in sets:
            s.setdefault('level', 50)
            if not s.get('evs'): s['evs'] = spread_for(rng, s['species'], s.get('nature'))
        teams.append({'source': 'research:' + (player or '')[:40], 'sets': sets})
    res = subprocess.run(['node', str(ROOT / 'search/validate_many.cjs')], input=json.dumps({'format': FMT, 'teams': [t['sets'] for t in teams]}),
                         capture_output=True, text=True, check=True)
    errs = json.loads(res.stdout)
    ok = [t for t, e in zip(teams, errs) if not e]
    for t, e in zip(teams, errs):
        if e: print('INVALID', t['source'], e[:2])
    print('teams', len(teams), 'valid', len(ok))
    Path(out).write_text(json.dumps(ok, indent=0))

main(sys.argv[1])
