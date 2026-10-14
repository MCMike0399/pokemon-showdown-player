"""Fetch public Showdown replays of one format into local JSONL shards (resumable, polite).

    .venv/bin/python search/human/fetch_replays.py --format gen9championsvgc2026regmc \
        --out data/replays/regmc [--min-rating 0] [--rate 2] [--max 80000]

Pages `replay.pokemonshowdown.com/search.json?format=F&before=T` backwards in time
(51 per page) and downloads each replay's JSON. Shards: out/replays-<k>.jsonl (one
JSON per line: id, uploadtime, rating, players, log); out/index.json records the
ids already stored and the oldest uploadtime reached, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = 'https://replay.pokemonshowdown.com'
UA = 'pokemon-showdown-player research fetcher (contact via github MCMike0399)'


def get(url: str, retries: int = 5) -> bytes | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            time.sleep(2 + 3 * attempt)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 + 3 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', default='gen9championsvgc2026regmc')
    ap.add_argument('--out', type=Path, default=Path('data/replays/regmc'))
    ap.add_argument('--rate', type=float, default=2.0, help='requests per second')
    ap.add_argument('--max', type=int, default=80000)
    ap.add_argument('--shard-size', type=int, default=1000)
    ap.add_argument('--before', type=int, help='start paging before this uploadtime (default: resume)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / 'index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'ids': [], 'oldest': None, 'done': False}
    have = set(index['ids'])
    before = args.before or index.get('oldest')
    shard_no = len(list(args.out.glob('replays-*.jsonl')))
    shard = args.out / f'replays-{shard_no:04d}.jsonl'
    in_shard = sum(1 for _ in shard.open()) if shard.exists() else 0
    delay = 1.0 / args.rate
    fetched = 0
    t0 = time.time()

    def save_index():
        index['ids'] = sorted(have)
        tmp = index_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(index))
        tmp.replace(index_path)

    while fetched < args.max:
        url = f'{BASE}/search.json?format={args.format}' + (f'&before={before}' if before else '')
        raw = get(url)
        time.sleep(delay)
        if raw is None:
            print(json.dumps({'error': 'search failed', 'url': url}), flush=True)
            break
        page = json.loads(raw)
        if not page:
            index['done'] = True
            print(json.dumps({'done': True, 'oldest': before}), flush=True)
            break
        for item in page:
            rid = item['id']
            before = min(before or item['uploadtime'], item['uploadtime'])
            if rid in have:
                continue
            body = get(f'{BASE}/{rid}.json')
            time.sleep(delay)
            if body is None:
                continue
            try:
                rep = json.loads(body)
            except ValueError:
                continue
            rec = {'id': rid, 'uploadtime': rep.get('uploadtime'), 'rating': rep.get('rating'),
                   'players': rep.get('players'), 'formatid': rep.get('formatid'), 'log': rep.get('log', '')}
            if in_shard >= args.shard_size:
                shard_no += 1
                shard = args.out / f'replays-{shard_no:04d}.jsonl'
                in_shard = 0
            with shard.open('a') as fh:
                fh.write(json.dumps(rec) + '\n')
            in_shard += 1
            have.add(rid)
            fetched += 1
            if fetched % 50 == 0:
                index['oldest'] = before
                save_index()
                rate = fetched / max(1e-9, time.time() - t0)
                print(json.dumps({'fetched': fetched, 'total': len(have), 'oldest': before,
                                  'oldest_iso': time.strftime('%Y-%m-%d %H:%M', time.gmtime(before)),
                                  'per_s': round(rate, 2)}), flush=True)
        index['oldest'] = before
        save_index()
    index['oldest'] = before
    save_index()
    print(json.dumps({'fetched': fetched, 'total': len(have), 'oldest': before}), flush=True)


if __name__ == '__main__':
    main()
