import json
from pathlib import Path
from ml.cluster import background_budget
from scripts.cluster_priority import needs_headroom, is_offline_leader, reconcile


def test_budget_is_opt_in_and_stale_data_defers(tmp_path, monkeypatch):
    monkeypatch.delenv('BURBUJA_BACKGROUND_BUDGET', raising=False)
    assert background_budget() is None
    file = tmp_path/'budget.json'
    assert background_budget(file, now=100) == 0
    for at, workers, expected in [(100,4,4),(9,4,0),(106,4,0),(100,True,0),(100,-1,0)]:
        file.write_text(json.dumps({'at':at,'maxWorkers':workers}))
        assert background_budget(file,now=100) == expected


def test_only_registered_offline_campaign_identity_is_eligible():
    root = Path('/home/user/pokemon')
    cmd = ['/bin/python','/home/user/pokemon.c10/search/campaign.py','worker','--dir','/home/user/pokemon/data/ml/value/campaign-c10']
    assert is_offline_leader(cmd,root)
    assert not is_offline_leader(['/bin/python','/home/user/pokemon.live/scripts/browser_play.py'],root)
    assert not is_offline_leader([*cmd[:-1],'/another/project'],root)
    assert not is_offline_leader([arg.replace('worker','run') for arg in cmd],root)


def test_priority_tracks_only_owned_identities_and_restores():
    calls = []
    setter = lambda pid, created, low: calls.append((pid,created,low)) or True
    owned = reconcile({}, {'42':100}, True,setter)
    assert owned == {'42':100}
    assert reconcile(owned, {'42':100}, True,setter) == owned
    assert calls == [('42',100,True)]
    assert reconcile(owned, {'42':200}, False,setter) == {}
    # Actual setter checks create_time before applying; reuse cannot affect a new process.
    assert calls[-1] == ('42',100,False)


def test_transcription_or_games_take_priority_but_stale_status_releases():
    state = {'at':100,'worker':{'active':'transcription'}}
    assert needs_headroom(state,100)
    assert not needs_headroom(state,191)
    assert needs_headroom({'at':100,'realtimeBusy':True},100)
    assert not needs_headroom({'at':100,'nodes':[]},100)
