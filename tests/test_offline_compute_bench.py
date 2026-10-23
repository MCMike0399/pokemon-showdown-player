"""Recovery may resume only the process identities the benchmark parked."""
from search import compute_bench as bench


def test_restore_does_not_signal_reused_campaign_pid(tmp_path, monkeypatch):
    bench.atomic_json(tmp_path/'lease.json', {
        'groups': [[100, 'search/campaign.py worker'], [101, 'search/campaign.py worker']],
        'supervisor': [102, 'search/campaign.py run'], 'benchmark_pids': [103]})
    current = {100: 'unrelated service', 101: 'search/campaign.py worker',
               102: 'search/campaign.py run', 103: 'unrelated service'}
    monkeypatch.setattr(bench, 'command', lambda p: current[p])
    monkeypatch.setattr(bench, 'PAUSE_MARKER', tmp_path/'absent')
    signals, registrations = [], []
    monkeypatch.setattr(bench.os, 'killpg', lambda p,s: signals.append(('group',p,s)))
    monkeypatch.setattr(bench.os, 'kill', lambda p,s: signals.append(('pid',p,s)))
    monkeypatch.setattr(bench, 'registry', lambda **kw: registrations.append(kw))
    bench.restore(tmp_path)
    assert [(kind,p) for kind,p,s in signals] == [('group',101),('pid',102)]
    assert registrations == [{'add':[101], 'remove':[103]}]
    bench.restore(tmp_path)
    assert len(signals) == 2  # idempotent after a successful restore


def test_restore_keeps_gates_paused_during_live_decision(tmp_path, monkeypatch):
    bench.atomic_json(tmp_path/'lease.json', {'groups': [[100, 'search/campaign.py worker']]})
    marker = tmp_path/'live-deciding'; marker.touch()
    monkeypatch.setattr(bench, 'PAUSE_MARKER', marker)
    monkeypatch.setattr(bench, 'command', lambda p: 'search/campaign.py worker')
    monkeypatch.setattr(bench, 'registry', lambda **kw: None)
    monkeypatch.setattr(bench.os, 'killpg', lambda *args: (_ for _ in ()).throw(AssertionError('unpaused live')))
    bench.restore(tmp_path)
    assert bench.read(tmp_path/'lease.json')['restored']
