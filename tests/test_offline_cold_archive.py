import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from ml.storage import fingerprint
from scripts import archive_collection_inputs as archive


@pytest.fixture
def closed_snapshot(tmp_path, monkeypatch):
    root=tmp_path/'ml';root.mkdir()
    with sqlite3.connect(root/'experience.sqlite3') as db:
        db.execute('CREATE TABLE jobs(id TEXT,kind TEXT,status TEXT,attempts INTEGER)')
        db.execute("INSERT INTO jobs VALUES('practice-fixture','practice','complete',1)")
    folder=root/'collections'/fingerprint(['practice-fixture',0]);(folder/'inputs').mkdir(parents=True)
    (folder/'plan.json').write_text('{}');(folder/'report.json').write_text('{}')
    source=folder/'inputs/experience.sqlite3'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE public_snapshot(value TEXT)')
        db.execute('INSERT INTO public_snapshot VALUES(?)',('retained evidence '*2000,))
    monkeypatch.setattr(archive,'opened_paths',lambda paths:set())
    return root,source


def test_archive_and_restore_preserve_every_byte_and_valid_sqlite(closed_snapshot):
    root,source=closed_snapshot;original=source.read_bytes()
    result=archive.run(root,True,16)
    assert len(result['archives'])==1 and not source.exists()
    zipped=Path(str(source)+'.gz');meta=json.loads(Path(str(source)+'.archive.json').read_text())
    assert meta['byte_exact_verified'] and meta['sha256']==hashlib.sha256(original).hexdigest()
    assert zipped.stat().st_size<len(original)
    archive.restore(zipped)
    assert source.read_bytes()==original and zipped.exists()
    with sqlite3.connect(source) as db:
        assert db.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        assert db.execute('SELECT COUNT(*) FROM public_snapshot').fetchone()[0]==1


def test_live_job_and_open_reader_exclude_snapshots(closed_snapshot,monkeypatch):
    root,source=closed_snapshot
    with sqlite3.connect(root/'experience.sqlite3') as db:db.execute("UPDATE jobs SET status='running'")
    assert archive.run(root,True,16)['archives']==[] and source.exists()
    with sqlite3.connect(root/'experience.sqlite3') as db:db.execute("UPDATE jobs SET status='complete'")
    monkeypatch.setattr(archive,'opened_paths',lambda paths:{str(source.resolve())})
    assert archive.run(root,True,16)['archives']==[] and source.exists()


def test_nonempty_journal_and_missing_report_exclude_snapshots(closed_snapshot):
    root,source=closed_snapshot;journal=Path(str(source)+'-journal');journal.write_bytes(b'in flight')
    assert archive.eligible(root)==[]
    journal.unlink();(source.parent.parent/'report.json').unlink()
    assert archive.eligible(root)==[] and source.exists()


def test_failure_during_roundtrip_keeps_original(closed_snapshot,monkeypatch):
    root,source=closed_snapshot;original=source.read_bytes()
    monkeypatch.setattr(archive,'digest',lambda reader:('bad hash',0))
    with pytest.raises(ValueError,match='roundtrip'):archive.archive(source,'practice-fixture',root)
    assert source.read_bytes()==original and not Path(str(source)+'.gz').exists()
    assert not list(source.parent.glob('.archive-*'))


def test_job_that_resumes_during_compression_keeps_original(closed_snapshot,monkeypatch):
    root,source=closed_snapshot;original=source.read_bytes();digest=archive.digest
    def resume(reader):
        value=digest(reader)
        with sqlite3.connect(root/'experience.sqlite3') as db:db.execute("UPDATE jobs SET status='running'")
        return value
    monkeypatch.setattr(archive,'digest',resume)
    with pytest.raises(ValueError,match='terminal'):archive.archive(source,'practice-fixture',root)
    assert source.read_bytes()==original and not Path(str(source)+'.gz').exists()


def test_restore_rejects_corruption_and_refuses_overwrite(closed_snapshot):
    root,source=closed_snapshot;archive.run(root,True,16);zipped=Path(str(source)+'.gz')
    metadata=Path(str(source)+'.archive.json');value=json.loads(metadata.read_text());good=value['sha256']
    value['sha256']='bad hash';metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='verification'):archive.restore(zipped)
    assert not source.exists() and zipped.exists()
    value['sha256']=good;metadata.write_text(json.dumps(value));archive.restore(zipped)
    with pytest.raises(ValueError,match='overwrite'):archive.restore(zipped)


def test_scheduled_archival_defers_to_resource_guard(closed_snapshot,monkeypatch):
    from ml.resources import ResourcePolicy
    root,source=closed_snapshot
    monkeypatch.setattr(ResourcePolicy,'sample',lambda self:{'training_allowed':False})
    result=archive.run(root,True,16,respect_budget=True)
    assert result['deferred'] and result['archives']==[] and source.exists()


def _candidate(root, name, finished=True):
    folder=root/'candidates'/name;(folder/'evaluation-inputs').mkdir(parents=True)
    if finished:(folder/'report.json').write_text(json.dumps({'evaluation':{'complete':True}}))
    source=folder/'evaluation-inputs/experience.sqlite3'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE t(v TEXT)');db.execute('INSERT INTO t VALUES(?)',('frozen input '*3000,))
    return source


def test_candidate_snapshots_compact_only_under_storage_pressure(closed_snapshot,monkeypatch):
    root,source=closed_snapshot
    done=_candidate(root,'finished');running=_candidate(root,'running',finished=False)
    original=done.read_bytes()
    monkeypatch.setattr('ml.continuous.LearningConfig.load',lambda r:type('C',(),{'resource':{'max_disk_gb':150.0}})())
    # Below the trigger: only completed collections are archived, candidates untouched.
    monkeypatch.setattr(archive,'data_gb',lambda r:100.0)
    result=archive.run(root,True,16,when_over=.85,target=.75)
    assert not result['storage_pressure'] and done.exists() and not source.exists()
    # Over the trigger: finished candidate snapshots compact losslessly; running ones never.
    monkeypatch.setattr(archive,'data_gb',lambda r:140.0)
    result=archive.run(root,True,16,when_over=.85,target=.75)
    assert result['storage_pressure'] and not done.exists() and running.exists()
    archive.restore(Path(str(done)+'.gz'))
    assert done.read_bytes()==original


def test_deferred_candidate_report_is_not_a_finished_evaluation(closed_snapshot):
    root, _ = closed_snapshot
    source = _candidate(root, 'paused')
    (source.parent.parent / 'report.json').write_text(json.dumps({'deferred': True, 'evaluation_progress': {'incumbent': 8}}))
    assert archive.eligible_candidates(root) == []


def test_shared_inference_directory_is_not_cold_archived(closed_snapshot):
    root, source = closed_snapshot
    original = source.parent
    pooled = root / 'input-snapshots' / 'pooled'
    pooled.parent.mkdir()
    original.rename(pooled)
    original.symlink_to(pooled, target_is_directory=True)
    assert archive.eligible(root) == []
