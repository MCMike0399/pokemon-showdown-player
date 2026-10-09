"""Download Smogon ladder usage statistics (chaos JSON) used as opponent set priors.

    .venv/bin/python search/fetch_usage.py [--month 2026-09] [--format gen9championsvgc2026regmc]

Files land in cache/usage/ (gitignored). The search agent reads the 0 cutoff by default,
which matches the whole ladder population rather than only high-rated players.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--month', default='2026-09')
    p.add_argument('--format', default='gen9championsvgc2026regmc')
    p.add_argument('--cutoffs', default='0,1500')
    args = p.parse_args()
    out = ROOT / 'cache' / 'usage'
    out.mkdir(parents=True, exist_ok=True)
    for cutoff in args.cutoffs.split(','):
        name = f'{args.format}-{cutoff}.json'
        url = f'https://www.smogon.com/stats/{args.month}/chaos/{name}'
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        (out / name).write_bytes(data)
        print(f'{url} -> {out / name} ({len(data)} bytes)')


if __name__ == '__main__':
    main()
