"""Account for retained model data and reserve headroom for live recordings.

The existing max_disk_gb configuration is in GiB. File sizes are conservative
logical bytes, counted once per inode; symlinks are never followed.
"""
from __future__ import annotations

import os
from pathlib import Path


def storage_roots(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if root.name == 'ml' and root.parent.name == 'data':
        repo = root.parent.parent
        return [root.parent, repo / 'artifacts', repo / 'cache']
    return [root]


def retained_bytes(root: Path) -> int:
    seen = set()
    total = 0
    for base in storage_roots(root):
        for folder, _, files in os.walk(base, followlinks=False):
            for name in files:
                path = Path(folder) / name
                try:
                    if path.is_symlink():
                        continue
                    stat = path.stat()
                except FileNotFoundError:
                    continue  # An atomic save/archive completed during the scan.
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    total += stat.st_size
                    seen.add(identity)
    return total


def storage_status(root: Path, max_disk_gb: float) -> dict:
    used = retained_bytes(root)
    budget = int(max_disk_gb * 2**30)
    # Stop optional production at 90%; allow the active game to finalize.
    background_limit = int(budget * .9)
    return {'retained_bytes': used, 'budget_bytes': budget,
            'background_limit_bytes': background_limit,
            'live_reserve_bytes': budget - background_limit,
            'background_allowed': used < background_limit,
            'budget_exceeded': used >= budget,
            'roots': [str(p) for p in storage_roots(root)]}
