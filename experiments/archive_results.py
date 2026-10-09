"""Preserve audited offline evidence with byte-verified compressed databases.

Complete-game shards are omitted only after an exact canonical episode audit.
Collecting checkpoints are restored at each referring run's relative archive path.
The archive is private data; never add its output to Git.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def verify(folder, integrity=False):
    folder=Path(folder).resolve()
    manifest=json.loads((folder/'archive-manifest.json').read_text())
    checked=databases=0
    for name, entry in manifest['files'].items():
        path=folder/name
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
            raise ValueError('archive transfer differs: '+name)
        if entry.get('encoding')=='gzip-sqlite':
            h=hashlib.sha256(); size=0
            with tempfile.TemporaryDirectory(prefix='offline-archive-check-') as tmp:
                restored=Path(tmp)/'experience.sqlite3'
                with gzip.open(path,'rb') as src, restored.open('wb') as dst:
                    for chunk in iter(lambda:src.read(2**20),b''):
                        h.update(chunk);size+=len(chunk)
                        if integrity:dst.write(chunk)
                if size!=entry['original_bytes'] or h.hexdigest()!=entry['original_sha256']:
                    raise ValueError('database is not byte-exact: '+name)
                if integrity:
                    c=sqlite3.connect(restored.as_uri()+'?mode=ro',uri=True)
                    try:
                        if c.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                            raise ValueError('SQLite integrity failed: '+name)
                    finally:c.close()
            databases+=1
        checked+=1
    return {'verified_files':checked,'byte_verified_databases':databases,
            'sqlite_integrity_checked':integrity,'retained_bytes':sum(x['bytes'] for x in manifest['files'].values())}

def restore_database(folder, database):
    folder=Path(folder).resolve()
    target=(folder/database).resolve()
    target.relative_to(folder)
    manifest=json.loads((folder/'archive-manifest.json').read_text())
    name=str(Path(database))+'.gz'
    entry=manifest['files'][name]
    if entry.get('encoding')!='gzip-sqlite':raise ValueError('not an archived SQLite database')
    if target.exists():
        if target.stat().st_size!=entry['original_bytes'] or sha(target)!=entry['original_sha256']:
            raise ValueError('existing restored database differs; refusing overwrite')
        return target
    source=folder/name
    if sha(source)!=entry['sha256']:raise ValueError('compressed database differs')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as out:
        temporary=Path(out.name)
        with gzip.open(source,'rb') as stream:shutil.copyfileobj(stream,out)
    try:
        if temporary.stat().st_size!=entry['original_bytes'] or sha(temporary)!=entry['original_sha256']:
            raise ValueError('restored bytes differ')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    return target

def build(output, audit_path, workers=4):
    output=Path(output).resolve()
    audit=json.loads(Path(audit_path).read_text())
    if audit['issues']:raise ValueError('source canonical/shard audit must pass')
    output.mkdir(parents=True,exist_ok=False)
    planned={}; checkpoints={}; canonical={}
    for group in audit['matrices']:
        run=Path(group['run'])
        canonical[str(run/'experience.sqlite3')]=ROOT/group['canonical_database']
        for path in (ROOT/run).glob('*'):
            if path.is_file() and path.suffix in ('.json','.py','.pt'):
                planned[str(path.relative_to(ROOT))]=path
        for path in (ROOT/run/'models').glob('*.pt'):
            planned[str(path.relative_to(ROOT))]=path
        c=sqlite3.connect((ROOT/group['canonical_database']).as_uri()+'?mode=ro',uri=True)
        try:
            for row in c.execute('SELECT DISTINCT json_extract(data,\'$.collecting_policy.checkpoint_sha256\') FROM episodes'):
                entry=audit['checkpoints'][row[0]]
                checkpoints[str(run/entry['canonical_archive_path'])]=ROOT/entry['source']
        finally:c.close()
    for matrix in sorted({str(Path(g['run']).parent) for g in audit['matrices']}):
        for path in (ROOT/matrix).glob('*'):
            if path.is_file() and path.suffix in ('.json','.py'):
                planned[str(path.relative_to(ROOT))]=path
    for directory in ('data/offline-corpus','artifacts/offline-analysis','artifacts/offline-scout',
                      'artifacts/offline-scout-comparison','artifacts/offline-worker-benchmark',
                      'artifacts/offline-training-benchmark','artifacts/offline-training-benchmark-large',
                      'artifacts/offline-training-benchmark-final','artifacts/offline-source-snapshot',
                      'artifacts/offline-source-snapshot-final'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
                name=str(path.relative_to(ROOT))
                if path.suffix=='.sqlite3':canonical[name]=path
                elif path.name.endswith(('-wal','-shm')):continue
                else:planned[name]=path
    notebook=ROOT/'artifacts/offline-notebook-executed.ipynb'
    if notebook.exists():planned[str(notebook.relative_to(ROOT))]=notebook
    # Do not retain repeated benchmark candidate checkpoints.
    planned={n:p for n,p in planned.items() if not ('offline-training-benchmark' in n and '/models/' in n)}
    files={}; hash_targets={}
    for name,source in {**planned,**checkpoints}.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        digest=sha(source)
        if digest in hash_targets:
            os.link(hash_targets[digest],target)
        else:
            shutil.copy2(source,target);hash_targets[digest]=target
        files[name]={'bytes':target.stat().st_size,'sha256':digest}
    def compress(item):
        name,source=item
        wal=Path(str(source)+'-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('close source SQLite writer before archive: '+name)
        target=output/(name+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256();size=0
        with source.open('rb') as src,target.open('wb') as out:
            with gzip.GzipFile(filename='',mode='wb',fileobj=out,compresslevel=6,mtime=0) as dst:
                for chunk in iter(lambda:src.read(2**20),b''):
                    h.update(chunk);size+=len(chunk);dst.write(chunk)
        return name+'.gz',{'bytes':target.stat().st_size,'sha256':sha(target),'encoding':'gzip-sqlite',
                         'original_bytes':size,'original_sha256':h.hexdigest()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for name,entry in pool.map(compress,canonical.items()):files[name]=entry
    manifest={'schema':'offline-evidence-archive-v1','files':files,'canonical_runs':audit['matrices'],
        'unique_collecting_checkpoints':audit['unique_collecting_checkpoints'],
        'canonical_episodes':sum(g['canonical_episodes'] for g in audit['matrices']),
        'omitted':'complete-run duplicate shard databases; per-game duplicate policies; temporary benchmark fits; Jupyter authentication/runtime/cache',
        'restore':'gzip -dk PATH/experience.sqlite3.gz; original hash and size must match this manifest before opening',
        'checkpoint_layout':'each canonical run retains models/collected/<format>/<sha>.pt at recorded relative paths',
        'source_audit':str(Path(audit_path).relative_to(ROOT))}
    (output/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verify(output),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','verify','restore'))
    p.add_argument('folder',type=Path)
    p.add_argument('--audit',type=Path,default=ROOT/'artifacts/offline-analysis/archive-input-audit.json')
    p.add_argument('--integrity',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--database',type=Path)
    a=p.parse_args()
    if a.action=='build':build(a.folder,a.audit,a.workers)
    elif a.action=='restore':
        if a.database is None:p.error('--database is required')
        print(restore_database(a.folder,a.database))
    else:print(json.dumps(verify(a.folder,a.integrity),indent=2))
