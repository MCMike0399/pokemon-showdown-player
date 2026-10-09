"""Bounded, resource-aware maintenance for the retained model-data service."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.archive_collection_inputs import run
from ml.storage import now


def maintain(root, limit=16):
    os.nice(15)
    with (root / 'cold-input-archive.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'checked_at': now(), 'skipped': True, 'reason': 'maintenance already running'}
        compression = run(root, True, limit, respect_budget=True, include_candidates=True)
        return {'checked_at': now(), 'compression': compression}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/ml')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('limit must be nonnegative')
    result = maintain(args.root, args.limit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.report)
    print(json.dumps({key: {k: v for k, v in value.items() if k != 'archives'} if isinstance(value, dict) else value
                      for key, value in result.items()}))
