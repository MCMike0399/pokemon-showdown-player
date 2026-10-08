import json

import pytest


def test_live_scout_uses_a_species_range_without_changing_predictions(tmp_path):
    from ml.storage import Store
    from ml.features import Features
    from ml.scout import Scout
    fmt = 'gen9championsvgc2026regmc'
    store = Store(tmp_path)
    # Exercise the actual predict() query against an older schema, then reopen
    # through the additive migration. The old full-table scan must go away.
    store.db.execute('DROP INDEX IF EXISTS scout_species_moves')
    moves = ['protect', 'tackle', 'ironhead']
    store.db.executemany('INSERT INTO scout_samples VALUES (?,?,?,?,?,?)',
                         [(str(i), fmt, 'kingambit', m, '[]', 'fixture') for i, m in enumerate(moves)])
    store.db.commit()
    features = Features({'moves': {m: {} for m in moves}, 'pokedex': {'kingambit': {'types': ['Steel', 'Dark'], 'baseStats': {'spe': 50}}}})
    scout = Scout(store, fmt, features);scout.trained_samples = 8
    log = ['|switch|p1a: A|Kingambit, L50|100/100', '|switch|p2a: B|Kingambit, L50|100/100', '|turn|1']
    queries = []
    store.db.set_trace_callback(queries.append)
    before = scout.predict(log, 'p1')
    query = next(q for q in queries if 'SELECT DISTINCT move FROM scout_samples' in q)
    old_plan = [r[3] for r in store.db.execute('EXPLAIN QUERY PLAN ' + query)]
    assert any('SCAN scout_samples' in p for p in old_plan)
    store.db.set_trace_callback(None)
    store.close()
    migrated = Store(tmp_path)
    new_plan = [r[3] for r in migrated.db.execute('EXPLAIN QUERY PLAN ' + query)]
    assert not any('SCAN scout_samples' in p for p in new_plan), new_plan
    assert any('COVERING INDEX' in p for p in new_plan), new_plan
    scout.store = migrated
    after = scout.predict(log, 'p1')
    assert len(before) == len(after) == 1
    a = {m['move']: m['probability'] for m in before[0]['moves']}
    b = {m['move']: m['probability'] for m in after[0]['moves']}
    assert b == pytest.approx(a, abs=1e-7)
    migrated.close()
