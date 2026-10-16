import importlib.util
import plistlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('live_deploy', ROOT / 'scripts' / 'live_deploy.py')
live_deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_deploy)


def git(cwd, *args):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    main = tmp_path / 'proj'
    main.mkdir()
    git(main, 'init', '-q', '-b', 'main')
    git(main, 'config', 'user.email', 't@t')
    git(main, 'config', 'user.name', 't')
    (main / 'a.txt').write_text('1')
    (main / '.gitignore').write_text('data/\n.env\n')
    git(main, 'add', '.')
    git(main, 'commit', '-q', '-m', 'one')
    (main / 'data').mkdir()
    (main / '.env').write_text('X=1')
    live = tmp_path / 'proj.live'
    monkeypatch.setattr(live_deploy, 'MAIN', main)
    monkeypatch.setattr(live_deploy, 'LIVE', live)
    return main, live


def test_pin_runs_an_exact_commit_and_shares_runtime_inputs(repo):
    main, live = repo
    first = live_deploy.pin('HEAD')
    # Work in progress in the main checkout never reaches the live worktree.
    (main / 'a.txt').write_text('half-finished')
    assert (live / 'a.txt').read_text() == '1'
    assert (live / 'data').resolve() == (main / 'data').resolve() and (live / '.env').is_symlink()
    (main / 'a.txt').write_text('2')
    git(main, 'commit', '-q', '-am', 'two')
    second = live_deploy.pin('HEAD')
    assert second != first and (live / 'a.txt').read_text() == '2'
    assert live_deploy.status()['dirty'] is False


def test_pin_refuses_a_hand_edited_live_worktree(repo):
    main, live = repo
    live_deploy.pin('HEAD')
    (live / 'a.txt').write_text('edited by hand')
    with pytest.raises(RuntimeError, match='tracked changes'):
        live_deploy.pin('HEAD')


def test_install_points_the_launch_agent_at_the_live_worktree(repo, tmp_path):
    main, live = repo
    plist = tmp_path / 'agent.plist'
    plist.write_bytes(plistlib.dumps({'ProgramArguments': ['/py', '-u', str(main / 'scripts/browser_play.py'), '--search'],
                                      'WorkingDirectory': str(main)}))
    assert live_deploy.install(plist, live) is True
    data = plistlib.loads(plist.read_bytes())
    assert data['ProgramArguments'] == ['/py', '-u', str(live / 'scripts/browser_play.py'), '--search']
    assert data['WorkingDirectory'] == str(live) and live_deploy.plist_points_live(plist, live)
    assert live_deploy.install(plist, live) is False  # idempotent
    assert list(tmp_path.glob('agent.plist.bak-*'))
