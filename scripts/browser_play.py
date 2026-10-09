#!/usr/bin/env python3
"""Bounded browser games: local PyTorch inference, Playwright clicks, no OpenClaw."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def credentials():
    values = {}
    env = HERE / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('\"\'')
    return tuple(os.environ.get(key) or values.get(key) for key in ('PS_USERNAME', 'PS_PASSWORD'))


async def run(args):
    from playwright.async_api import async_playwright
    from ml.brain import Brain
    from ml.browser import BrowserPlayer
    from ml.storage import Store
    from ml.reload import pin_runtime, source_generation

    pin_runtime(source_generation())

    args.root.mkdir(parents=True, exist_ok=True)
    # Share ownership with the former protocol controller. No competing player.
    (HERE / 'data/ml').mkdir(parents=True, exist_ok=True)
    owner = (HERE / 'data/ml/ladder-owner.lock').open('a')
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close()
        raise ValueError('another ladder controller owns this account; stop it first')
    try:
        checkpoint = args.root / 'models' / (args.format + '.pt')
        if not args.inspect and not checkpoint.exists():
            raise ValueError('no selected local checkpoint for this format')
        team = None
        if args.team:
            team = json.loads(args.teams_file.read_text())[args.team]
            if team['format'] != args.format:
                raise ValueError('team format mismatch')
            team = {**team, 'name': args.team}
        if not args.inspect and not team and 'random' not in args.format:
            raise ValueError('team is required for matchmaking in this format')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = args.root / 'browser-runs' / stamp
        output.mkdir(parents=True, mode=0o700)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        try:
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch_persistent_context(
                    str(args.profile), headless=not args.headed,
                    viewport={'width': 1440, 'height': 900},
                    **({'channel': 'chrome'} if args.chrome else {}))
                try:
                    page = browser.pages[0] if browser.pages else await browser.new_page()
                    await page.goto('https://play.pokemonshowdown.com/', wait_until='domcontentloaded')
                    await page.wait_for_function('window.app && app.user && window.BattleFormats')
                    player = BrowserPlayer(page, None if args.inspect else Brain(Store(args.root)), log_path=output / 'events.jsonl')
                    agent = None
                    if args.search and not args.inspect:
                        from search.agent import SearchAgent
                        brain = player.brain

                        async def preview(ctx):
                            return brain.decide(ctx, explore=False, record=False)['choice']
                        agent = SearchAgent(fmt=args.format, worlds=args.search_worlds, engines=args.search_engines,
                                            screen={'probe': 4, 'keep_opp': 20, 'keep_ours': 24}, preview=preview,
                                            seed=int(time.time()))

                        async def search_decider(ctx):
                            return await agent.decide(ctx)
                        player.decider = search_decider
                        player.log('search_agent', worlds=args.search_worlds, engines=args.search_engines)
                    if args.inspect:
                        # Read-only transport smoke: no login, matchmaking or moves.
                        print(json.dumps(await page.evaluate("() => ({client:!!app.rooms, named:!!app.user.get('named'), title:document.title})")))
                        await page.screenshot(path=str(output / 'client.png'))
                        return
                    username, password = credentials()
                    if not username or not password:
                        raise ValueError('configured account credentials are required')
                    await player.login(username, password)
                    active = await page.evaluate("() => Object.values(app.rooms).filter(r => r.id.startsWith('battle-') && r.request && r.request.side && !r.battle.ended).map(r => r.id)")
                    if active and active != [args.room]:
                        raise ValueError('existing active battle; resume its exact room before searching')
                    for game in range(args.games):
                        if stop.is_set():
                            break
                        if game:
                            try:
                                await asyncio.wait_for(stop.wait(), timeout=args.game_delay)
                                break
                            except asyncio.TimeoutError:
                                pass
                        room = args.room if game == 0 else None
                        if room:
                            await page.goto('https://play.pokemonshowdown.com/' + room, wait_until='domcontentloaded')
                            await page.wait_for_function('id => {const r=app.rooms[id];return r && (r.request || (r.battle && r.battle.stepQueue.some(l => l.startsWith("|win|") || l === "|tie" || l === "|tie|")));}', arg=room)
                        else:
                            await player.search(args.format, team)
                            deadline = time.monotonic() + args.search_timeout
                            while time.monotonic() < deadline and not stop.is_set():
                                rooms = await page.evaluate("() => Object.values(app.rooms).filter(r => r.id.startsWith('battle-') && r.request && r.request.side && !r.battle.ended).map(r => r.id)")
                                if len(rooms) > 1:
                                    raise ValueError('multiple active battles; refuse ambiguous ownership')
                                if rooms:
                                    room = rooms[0]
                                    break
                                popup = page.locator('.ps-popup:visible')
                                if await popup.count():
                                    player.log('search_rejected')
                                    raise ValueError('server rejected matchmaking; inspect browser before retrying')
                                await asyncio.sleep(.25)
                            if not room:
                                await player.cancel_search()
                                if stop.is_set():
                                    break
                                raise TimeoutError('matchmaking timed out; search cancelled')
                        outcome = await player.play(room, args.format, sets=team['sets'] if team else None,
                                                    timeout=args.stall_timeout, record=not args.no_record)
                        print(json.dumps({'game': game + 1, 'result': outcome}), flush=True)
                    if agent is not None:
                        player.log('search_stats', **{k: v for k, v in agent.stats.items() if k != 'errors'},
                                   errors=agent.stats['errors'][-20:])
                        await agent.close()
                    player.log('stopped', finish_active_game=True)
                    print('Browser run stopped; evidence: ' + str(output), flush=True)
                except Exception:
                    if 'player' in locals():
                        player.log('interrupted', room=locals().get('room'))
                    raise
                finally:
                    await browser.close()
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)
    finally:
        owner.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=HERE / 'data/ml')
    p.add_argument('--profile', type=Path, default=HERE / 'data/browser-profile')
    p.add_argument('--format', default='gen9championsvgc2026regmc')
    p.add_argument('--team')
    p.add_argument('--teams-file', type=Path, default=HERE / 'teams.json')
    p.add_argument('--room', help='Resume this existing room before any new search')
    p.add_argument('--games', type=int, default=1)
    p.add_argument('--headed', action='store_true')
    p.add_argument('--chrome', action='store_true', help='Use installed Google Chrome')
    p.add_argument('--inspect', action='store_true', help='Read-only browser check; no login or games')
    p.add_argument('--no-record', action='store_true', help='Greedy inference without PPO recording')
    p.add_argument('--search', action='store_true', help='Decide turns with the simulator search agent (implies --no-record)')
    p.add_argument('--search-worlds', type=int, default=6, help='Determinized opponent worlds per decision')
    p.add_argument('--search-engines', type=int, default=3, help='Parallel search engine processes')
    p.add_argument('--search-timeout', type=int, default=120)
    p.add_argument('--stall-timeout', type=int, default=180)
    p.add_argument('--game-delay', type=float, default=30, help='Seconds between completed games and the next search')
    return p


if __name__ == '__main__':
    p = parser()
    args = p.parse_args()
    if not 1 <= args.games <= 1000 or not 5 <= args.search_timeout <= 600 or not 5 <= args.stall_timeout <= 600:
        p.error('games must be 1..1000 and timeouts 5..600 seconds')
    if not 0 <= args.game_delay <= 3600:
        p.error('game delay must be 0..3600 seconds')
    if args.search:
        args.no_record = True
        if not 1 <= args.search_worlds <= 32 or not 1 <= args.search_engines <= 8:
            p.error('search worlds must be 1..32 and engines 1..8')
    if not args.format.isalnum() or (args.room and not all(c.isalnum() or c == '-' for c in args.room)):
        p.error('format and room must be Showdown identifiers')
    asyncio.run(run(args))
