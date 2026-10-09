"""Exploratory run with the unshrunk Reddit-derived completed-episode budgets."""
import asyncio
import json
import os
from pathlib import Path
from experiments.offline_ppo import ARMS, Compute, prepare_teams, run_matrix

if __name__ == '__main__':
    teams = asyncio.run(prepare_teams())['teams']
    path = Path('data/offline-corpus/macintosh-compute.json')
    tuned = json.loads(path.read_text()) if path.exists() else {}
    compute = Compute(workers=tuned.get('workers',max(1,(os.cpu_count() or 2)-1)),backend=tuned.get('backend','cpu'))
    run_matrix(Path('artifacts/offline-ppo-reddit-budget'),teams,compute,
               arms=ARMS,seeds=(0,1,2),updates=5,eval_games=32,eval_every=5)
