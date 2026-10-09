import json
import os

import pytest

from ml.data_budget import retained_bytes, storage_roots, storage_status
from ml.input_snapshot import freeze_inputs
from ml.storage import Store

FMT = 'gen9championsvgc2026regmc'


def test_budget_counts_experiments_once_and_skips_external_symlinks(tmp_path):
    root = tmp_path / 'repo/data/ml'
    root.mkdir(parents=True)
    (root / 'data').write_bytes(b'x' * 100)
    artifact = tmp_path / 'repo/artifacts'
    artifact.mkdir()
    os.link(root / 'data', artifact / 'same')
    (artifact / 'experiment').write_bytes(b'y' * 80)
    external = tmp_path / 'external'
    external.mkdir()
    (external / 'unrelated').write_bytes(b'z' * 200)
    (artifact / 'worktree').symlink_to(external, target_is_directory=True)
    (artifact / 'link').symlink_to(external / 'unrelated')
    assert len(storage_roots(root)) == 3
    assert retained_bytes(root) == 180
    # Test byte boundaries with a small budget, including a protected live reserve.
    state = storage_status(root, 200 / 2**30)
    assert not state['background_allowed'] and not state['budget_exceeded']
    assert state['live_reserve_bytes'] == 20


def test_shared_inputs_preserve_inference_and_isolate_live_updates(tmp_path):
    store = Store(tmp_path / 'live')
    try:
        with store.db:
            store.db.execute('INSERT INTO public_battles VALUES(?,?,?,?,?,?)',
                             ('battle', FMT, 'external', 'date', 'digest', '["retained replay"]'))
            store.db.execute('INSERT INTO scout_samples VALUES(?,?,?,?,?,?)',
                             ('sample', FMT, 'pelipper', 'protect', '[1,2,3]', 'digest'))
            store.db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?,?)',
                             ('doc', 'url', FMT, 'title', 'date', 'date', 'original research', 'digest'))
        scouts = store.root / 'scouts'
        scouts.mkdir()
        (scouts / (FMT + '.pt')).write_bytes(b'frozen weights')
        first, second, third = [tmp_path / name for name in ('a', 'b', 'c')]
        freeze_inputs(store, first, FMT)
        freeze_inputs(store, second, FMT)
        assert first.resolve() == second.resolve()
        assert (first / 'scouts' / (FMT + '.pt')).read_bytes() == b'frozen weights'
        with pytest.raises(ValueError, match='immutable'):
            Store(first)
        frozen = Store.read_only(first)
        try:
            assert frozen.db.execute('SELECT DISTINCT move FROM scout_samples WHERE format=? AND species=?', (FMT, 'pelipper')).fetchone()[0] == 'protect'
            assert frozen.db.execute('SELECT text FROM documents').fetchone()[0] == 'original research'
            assert frozen.db.execute('SELECT vector FROM scout_samples').fetchone()[0] == '[]'
            assert frozen.db.execute('SELECT COUNT(*) FROM public_battles').fetchone()[0] == 0
            assert json.loads((first / 'inference-only.json').read_text())['training_allowed'] is False
            with store.db:
                store.db.execute("UPDATE scout_samples SET move='hurricane'")
            freeze_inputs(store, third, FMT)
            assert third.resolve() != first.resolve()
            assert frozen.db.execute('SELECT move FROM scout_samples').fetchone()[0] == 'protect'
            freeze_inputs(store, first, FMT)  # Retry retains its declared evidence.
            assert store.db.execute('SELECT vector FROM scout_samples').fetchone()[0] == '[1,2,3]'
            assert store.db.execute('SELECT log FROM public_battles').fetchone()[0] == '["retained replay"]'
        finally:
            frozen.close()
    finally:
        store.close()


def test_freeze_preserves_research_order_for_tied_species_priors(tmp_path):
    from ml.research import species_prior
    from ml.storage import now
    store = Store(tmp_path / 'live')
    try:
        with store.db:
            for key, prefix in [('z-first', 'First'), ('a-second', 'Second')]:
                sets = [{'species': prefix + str(i)} for i in range(30)]
                store.db.execute('INSERT INTO research_teams VALUES(?,?,?,?,?,?,?)',
                    (key, FMT, 'url', '', '', now(), json.dumps(sets)))
        destination = tmp_path / 'frozen'
        freeze_inputs(store, destination, FMT)
        frozen = Store.read_only(destination)
        try:
            assert species_prior(store, FMT) == species_prior(frozen, FMT)
        finally:
            frozen.close()
    finally:
        store.close()


def test_duplicate_move_evidence_does_not_duplicate_inference_inputs(tmp_path):
    store = Store(tmp_path / 'live')
    try:
        with store.db:
            store.db.execute('INSERT INTO scout_samples VALUES(?,?,?,?,?,?)',
                ('first', FMT, 'pelipper', 'protect', '[1]', 'battle-one'))
        first, second = tmp_path / 'first', tmp_path / 'second'
        freeze_inputs(store, first, FMT)
        with store.db:
            store.db.execute('INSERT INTO scout_samples VALUES(?,?,?,?,?,?)',
                ('second', FMT, 'pelipper', 'protect', '[2]', 'battle-two'))
        freeze_inputs(store, second, FMT)
        assert first.resolve() == second.resolve()
        assert store.db.execute('SELECT COUNT(*) FROM scout_samples').fetchone()[0] == 2
    finally:
        store.close()


def test_shared_input_link_works_through_a_directory_alias(tmp_path):
    store = Store(tmp_path / 'canonical')
    try:
        actual = tmp_path / 'actual/nested'
        actual.mkdir(parents=True)
        alias = tmp_path / 'alias'
        alias.symlink_to(actual, target_is_directory=True)
        destination = alias / 'inputs'
        freeze_inputs(store, destination, FMT)
        reader = Store.read_only(destination)
        reader.close()
        assert destination.resolve().parent == (store.root / 'input-snapshots').resolve()
    finally:
        store.close()
