"""Lossless, reversible archival of completed collection inference snapshots.

Under storage pressure (--when-over), finished candidates' frozen evaluation
input snapshots are compacted too. Never archives the main experience DB,
live/queued/running inputs, unfinished candidates or checkpoints. Every byte is round-trip verified before the
uncompressed duplicate is removed. Default is a dry run.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from ml.storage import fingerprint


def digest(reader):
    result=hashlib.sha256();size=0
    for chunk in iter(lambda:reader.read(1024*1024),b''):
        result.update(chunk);size+=len(chunk)
    return result.hexdigest(),size


def eligible(root):
    db=sqlite3.connect((root/'experience.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    jobs={}
    for job,status,attempts in db.execute("SELECT id,status,attempts FROM jobs WHERE kind='practice'"):
        for attempt in range(attempts+1):jobs[fingerprint([job,attempt])]=(job,status)
    result=[]
    for folder in sorted((root/'collections').iterdir()) if (root/'collections').exists() else []:
        source=folder/'inputs/experience.sqlite3';job=jobs.get(folder.name)
        if not job or job[1]!='complete' or not source.is_file() or source.is_symlink():continue
        if not (folder/'plan.json').is_file() or not (folder/'report.json').is_file():continue
        if any(Path(str(source)+suffix).exists() and Path(str(source)+suffix).stat().st_size for suffix in ['-wal','-journal']):continue
        result.append((source,job[0]))
    db.close();return result


def eligible_candidates(root):
    """Finished candidate evaluations: their frozen input DB snapshot is never written again."""
    result=[]
    for folder in sorted((root/'candidates').iterdir()) if (root/'candidates').exists() else []:
        source=folder/'evaluation-inputs/experience.sqlite3'
        if not source.is_file() or source.is_symlink() or not (folder/'report.json').is_file():continue
        if any(Path(str(source)+suffix).exists() and Path(str(source)+suffix).stat().st_size for suffix in ['-wal','-journal']):continue
        result.append((source,'candidate:'+folder.name))
    # Oldest first: the most recent evaluations are the likeliest to be inspected.
    result.sort(key=lambda item:item[0].stat().st_mtime)
    return result


def data_gb(root):
    out=subprocess.run(['du','-sk',str(root)],capture_output=True,text=True)
    return int(out.stdout.split()[0])/2**20 if out.returncode==0 and out.stdout else 0.0


def opened_paths(paths):
    executable=shutil.which('lsof')
    if not executable:raise RuntimeError('lsof required to verify snapshots are not open')
    result=subprocess.run([executable,'-F','pcn','--',*[str(p.resolve()) for p in paths]],capture_output=True,text=True)
    if result.returncode not in (0,1):raise RuntimeError('failed to inspect open file handles')
    # Apple's Docker file-sharing broker caches VFS handles to every accessed
    # path, including our fresh compression probe. These are not collection
    # consumers. Completed-job/report checks remain mandatory; any other open
    # process (including Python, Node or a replay reader) excludes the file.
    command='';opened=set()
    for line in result.stdout.splitlines():
        if line.startswith('p'):command=''
        elif line.startswith('c'):command=line[1:]
        elif line.startswith('n') and command!='com.apple.Virtualization.Virtua':opened.add(line[1:])
    return opened


def archive(source,job,root):
    destination=Path(str(source)+'.gz');metadata=Path(str(source)+'.archive.json')
    if destination.exists():return {'path':str(source),'skipped':'archive already exists'}
    before=source.stat()
    con=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        if con.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('invalid input snapshot')
    finally:con.close()
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=source.parent,prefix='.archive-',delete=False) as raw:
            temporary=Path(raw.name);hasher=hashlib.sha256();size=0
            with source.open('rb') as reader,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3) as zipped:
                for block in iter(lambda:reader.read(1024*1024),b''):
                    hasher.update(block);size+=len(block);zipped.write(block)
            raw.flush();os.fsync(raw.fileno())
        with gzip.open(temporary,'rb') as reader:verified,verified_size=digest(reader)
        if verified!=hasher.hexdigest() or verified_size!=size:raise ValueError('archive roundtrip failed')
        current=source.stat()
        if (before.st_ino,before.st_mtime_ns,before.st_size)!=(current.st_ino,current.st_mtime_ns,current.st_size):raise ValueError('snapshot changed during archive')
        if job.startswith('candidate:'):
            if not (source.parent.parent/'report.json').is_file():raise ValueError('candidate is no longer terminal')
        else:
            db=sqlite3.connect((root/'experience.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
            try:
                row=db.execute('SELECT status FROM jobs WHERE id=?',(job,)).fetchone()
                if not row or row[0]!='complete':raise ValueError('collection is no longer terminal')
            finally:db.close()
        if str(source.resolve()) in opened_paths([source]):raise ValueError('snapshot opened during archive')
        if any(Path(str(source)+suffix).exists() and Path(str(source)+suffix).stat().st_size for suffix in ['-wal','-journal']):raise ValueError('snapshot gained a journal')
        info={'original':source.name,'job':job,'sha256':verified,'original_bytes':size,'gzip_bytes':temporary.stat().st_size,'compression':'gzip-level3','byte_exact_verified':True}
        temporary.chmod(before.st_mode & 0o777)
        temporary.replace(destination);temporary=None
        temp_meta=metadata.with_suffix('.tmp')
        with temp_meta.open('w') as handle:
            json.dump(info,handle,indent=2);handle.flush();os.fsync(handle.fileno())
        temp_meta.replace(metadata)
        # Persist the verified archive and its restore metadata before removing
        # the duplicate, including across a host crash between these operations.
        directory=os.open(source.parent,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        source.unlink()
        return {'path':str(source),**info,'saved_bytes':size-destination.stat().st_size}
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def restore(zipped):
    source=Path(str(zipped)[:-3]);meta=json.load(open(Path(str(source)+'.archive.json')))
    if source.exists():raise ValueError('original already exists; never overwrite it')
    temporary=source.with_suffix('.restore-tmp')
    try:
        hasher=hashlib.sha256();size=0
        with gzip.open(zipped,'rb') as reader,temporary.open('xb') as writer:
            for block in iter(lambda:reader.read(1024*1024),b''):
                writer.write(block);hasher.update(block);size+=len(block)
            writer.flush();os.fsync(writer.fileno())
        if hasher.hexdigest()!=meta['sha256'] or size!=meta['original_bytes']:raise ValueError('restoration verification failed')
        temporary.replace(source)
    finally:temporary.unlink(missing_ok=True)
    return {'restored':str(source),'sha256':meta['sha256']}


def run(root,apply,limit,respect_budget=False,when_over=None,target=None):
    """Collections are always eligible. When data use exceeds `when_over` x budget,
    finished candidate snapshots are compacted too, until use falls below `target` x budget."""
    paths=eligible(root)
    budget=None;used=None;pressure=False
    if when_over is not None:
        from ml.continuous import LearningConfig
        budget=float(LearningConfig.load(root).resource['max_disk_gb']);used=data_gb(root)
        pressure=used>when_over*budget
        if pressure:paths=paths+eligible_candidates(root)
    opened=opened_paths([p for p,_ in paths]) if paths else set();paths=[(p,j) for p,j in paths if str(p.resolve()) not in opened]
    if limit:paths=paths[:limit]
    policy=None
    if respect_budget:
        from ml.continuous import LearningConfig
        from ml.resources import ResourcePolicy
        policy=ResourcePolicy(**LearningConfig.load(root).resource)
    result={'eligible':len(paths),'eligible_bytes':sum(p.stat().st_size for p,_ in paths),'applied':apply,'archives':[],
            'data_gb':None if used is None else round(used,3),'budget_gb':budget,'storage_pressure':pressure}
    if apply:
        freed=0
        for path,job in paths:
            if pressure and target is not None and job.startswith('candidate:') and used-freed/2**30<target*budget:
                result['reached_target']=True;break
            if policy:
                resources=policy.sample()
                if not resources['training_allowed']:
                    result.update(deferred=True,reason='host resource budget',resources=resources);break
            entry=archive(path,job,root)
            freed+=entry.get('saved_bytes',0)
            result['archives'].append(entry)
            print(json.dumps(result['archives'][-1]),flush=True)
        result['freed_gb']=round(freed/2**30,3)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'data/ml');p.add_argument('--apply',action='store_true');p.add_argument('--limit',type=int,default=0);p.add_argument('--respect-resource-budget',action='store_true');p.add_argument('--restore',type=Path);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--when-over',type=float,help='Also compact finished candidate snapshots when data use exceeds this fraction of max_disk_gb')
    p.add_argument('--target',type=float,default=.75,help='Stop candidate compaction below this fraction of max_disk_gb');a=p.parse_args()
    if a.when_over is not None and not 0<a.target<a.when_over<=1:p.error('need 0 < target < when-over <= 1')
    if a.limit<0:p.error('limit must be nonnegative')
    a.report.parent.mkdir(parents=True,exist_ok=True)
    os.nice(15)
    with (a.root/'cold-input-archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=restore(a.restore) if a.restore else run(a.root,a.apply,a.limit,a.respect_resource_budget,a.when_over,a.target if a.when_over is not None else None)
    a.report.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='archives'}))
