"""Read-only, readable inspection of exact decision snapshots in the local archive."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.recording import read_archive

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--alternatives', action='store_true')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'data/ml')
    args = parser.parse_args()
    print(json.dumps(read_archive(args.root, args.room, args.start, args.limit, args.alternatives), indent=2))
