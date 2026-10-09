"""Benchmark Macintosh offline workloads and train the real-battle scout separately."""
import argparse
import asyncio
import json
from pathlib import Path
from experiments.offline_ppo import (
    CORPUS, prepare_teams, prepare_scout, benchmark_workers, benchmark_training,
    compare_scout, Compute, FMT, write_json, inventory,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('workers','training','scout','scout-comparison'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=Path)
    parser.add_argument('--workers', type=int, nargs='+', default=[1,4,8,12,14])
    parser.add_argument('--games', type=int, default=28)
    parser.add_argument('--backend', choices=('cpu','mps'), default='cpu')
    args = parser.parse_args()
    if args.kind == 'training':
        if not args.episodes:
            parser.error('--episodes must point to a completed arm directory')
        result = benchmark_training(args.output,args.episodes)
    elif args.kind == 'scout':
        result = asyncio.run(prepare_scout(CORPUS,args.output,backend=args.backend))
    else:
        teams = asyncio.run(prepare_teams())['teams']
        if args.kind == 'workers':
            result = benchmark_workers(args.output,teams,counts=args.workers,count_games=args.games)
        else:
            result = compare_scout(args.output,teams,Compute(workers=max(args.workers),backend=args.backend),
                CORPUS/'models'/(FMT+'.pt'),Path('artifacts/offline-scout'),count=args.games)
    write_json(args.output/'hardware.json',inventory())
    print(json.dumps([{k:v for k,v in row.items() if k!='games' or not isinstance(v,list)}
                      for row in result], indent=2))

if __name__ == '__main__':
    main()
