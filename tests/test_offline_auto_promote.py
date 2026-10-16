"""scripts/auto_promote.py rewrites search/config.py LIVE without disturbing the rest of the file."""
import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('auto_promote', ROOT / 'scripts' / 'auto_promote.py')
auto_promote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_promote)


def live_of(src: str) -> dict:
    ns: dict = {}
    exec(compile(ast.parse(src), 'config', 'exec'), ns)
    return ns['LIVE']


def test_apply_options_replaces_and_adds_keys():
    src = (ROOT / 'search' / 'config.py').read_text()
    opts = {'alpha': 0.25, 'leaf': {'a': 0.801, 'b': -0.43}, 'speed_infer': True,
            'protect_d2': {'behind': -1.0, 'top_opp': 3, 'keep': 6, 'probes': 2}}
    out = auto_promote.apply_options(src, opts, 'test promotion')
    live, before = live_of(out), live_of(src)
    assert {k: live[k] for k in opts} == opts
    assert {k: v for k, v in live.items() if k not in opts} == {k: v for k, v in before.items() if k not in opts}
    assert out.split('LIVE = {')[0] == src.split('LIVE = {')[0]
    assert out.split('def agent_kwargs')[1] == src.split('def agent_kwargs')[1]
    # Idempotent on a second promotion of the same keys.
    assert live_of(auto_promote.apply_options(out, opts, 'again')) == live


def test_stage_state_skips_a_half_written_line(tmp_path):
    rows = [{'pair': 0, 'arm': 'cand', 'win': True, 'score': 1.0}, {'pair': 0, 'arm': 'inc', 'win': False, 'score': 0.0}]
    (tmp_path / 'x-stage.jsonl').write_text(''.join(__import__('json').dumps(r) + '\n' for r in rows) + '{"pair": 1, "ar')
    st = auto_promote.stage_state(tmp_path, 'x-stage', 1, 0.05)
    assert st['pairs_complete'] == 1 and st['gains'] == 1 and st['settled'] and not st['running']


def test_sync_workspace_leaves_another_branch_alone(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.run(['git', *args], cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    git('init', '-q', '-b', 'main')
    git('config', 'user.email', 't@t')
    git('config', 'user.name', 't')
    (tmp_path / 'a').write_text('1')
    git('add', '.')
    git('commit', '-q', '-m', 'one')
    git('checkout', '-q', '-b', 'ui-feature')
    calls = []
    monkeypatch.setattr(auto_promote, 'ROOT', tmp_path)
    monkeypatch.setattr(auto_promote, 'run', lambda gates, cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(auto_promote, 'log', lambda gates, msg: calls.append(msg))
    auto_promote.sync_workspace(tmp_path)
    assert not any(isinstance(c, list) and 'merge' in c for c in calls)
    assert git('rev-parse', '--abbrev-ref', 'HEAD') == 'ui-feature'
