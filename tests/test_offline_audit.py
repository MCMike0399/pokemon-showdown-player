import copy
import hashlib
import json
from pathlib import Path

from harness import TeamStore
from ml.brain import Brain
from ml.features import Features
from ml.storage import Store
from scripts.audit_campaign import audit
from test_offline_learning import context, FMT


def test_audit_keeps_fragment_outcome_stratifies_and_never_relabels(tmp_path):
    root = tmp_path / 'ml'
    store = Store(root)
    brain = Brain(store, Features())
    records = []
    for i, outcome in enumerate((1, -1)):
        ctx = context('local-' + str(i))
        ctx['team_id'] = 'team-' + str(i)
        brain.decide(ctx, explore=True, record=True)
        brain.finish(ctx['room'], {'winner': 'Player' if outcome == 1 else 'Opponent'}, 'Player')
        records.append({'room': ctx['room']})
    originals = store.episodes(FMT)
    legacy = copy.deepcopy(originals[0])
    legacy['steps'][0].pop('snapshot')
    # Separate incomplete fragment makes game2 unsuitable for trajectory analysis.
    fragment = copy.deepcopy(originals[1])
    fragment.update(id='fragment', status='abandoned', outcome=None)
    store.save_episode(fragment)
    with store.db:
        store.db.execute('UPDATE episodes SET data=?, trained=1 WHERE id=?', (json.dumps(legacy), legacy['id']))
    before = list(store.db.execute('SELECT id,status,trained,data FROM episodes ORDER BY id'))
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    (campaign / 'manifest.json').write_text(json.dumps({'format': FMT}))
    (campaign / 'games.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    teams = TeamStore(tmp_path / 'teams.json')
    checkpoint = root / 'models' / (FMT + '.pt')
    before_weights = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = audit(campaign, root, teams)
    assert result['summary']['games'] == 2
    assert result['summary']['wins'] == result['summary']['losses'] == 1
    assert result['summary']['usable_decision_episodes'] == 1
    assert len(result['strata']) == 2
    assert result['coverage']['legacy_encoded_decisions'] == 1
    assert result['excluded'][0]['reason'].startswith('fragmented')
    assert [tuple(r) for r in before] == [tuple(r) for r in store.db.execute('SELECT id,status,trained,data FROM episodes ORDER BY id')]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before_weights
    store.close()
