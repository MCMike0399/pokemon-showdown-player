#!/usr/bin/env python3
"""Live games: local model decisions; Playwright clicks (IPlayer default), OpenClaw's MCP harness as the fallback transport."""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import itertools
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


def save_screen_records(agent, team, room):
    """Keep a finished game's probe screening results as ranker labels; never raises."""
    records = agent.take_screen_records()  # always drained: a long run must not accumulate them
    if not team:
        return
    try:
        from search.screen_data import screen_root, write_records
        from search.value_service import load_config
        from search.value_store import dir_bytes
        root = screen_root(team)
        cfg = load_config()
        if records and room and cfg['screen_enabled'] and dir_bytes(root) <= cfg['screen_max_gb'] * 2**30:
            name = str(room).removeprefix('battle-')
            write_records(records, root / f'live-{name}.npz', f'live-{name}')
    except Exception as error:  # data logging must never cost a live game
        print(json.dumps({'screen_records_error': repr(error)[:300]}), flush=True)


def load_team(teams_file, name, fmt):
    team = json.loads(Path(teams_file).read_text())[name]
    if team['format'] != fmt:
        raise ValueError('team format mismatch')
    return {**team, 'name': name}


def arm_team(arm, spec, default):
    """The team a game plays: the arm's `team` override, else the A/B spec's base team
    (a resumed room the experiment does not know), else the runner's --team."""
    over = (arm or {}).get('overrides') or {}
    return over.get('team') or ((spec or {}).get('base') or {}).get('team') or default


def prepare_matchmaking(brain, fmt):
    """Adopt an evaluated candidate only before starting a new game."""
    from ml.promotion import promote_ready
    promotion = promote_ready(brain.store, fmt)
    if promotion.get('promoted'):
        brain.model(fmt)  # Reload the CPU checkpoint for the next preview.
    return promotion


async def run(args):
    from playwright.async_api import async_playwright
    from ml.brain import Brain
    from ml.browser import BrowserPlayer
    from ml.storage import Store
    from ml.continuous import LearningConfig
    from ml.data_budget import storage_status
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
            team = load_team(args.teams_file, args.team, args.format)
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
                    fallback = None
                    if args.fallback == 'mcp' and not args.inspect:
                        # Started only if Chrome fails to deliver a choice (ml/players.py).
                        from ml.players import McpPlayer
                        fallback = McpPlayer()
                    player = BrowserPlayer(page, None if args.inspect else Brain(Store(args.root)),
                                           log_path=output / 'events.jsonl', fallback=fallback)
                    if team:
                        # Every run and every result names its team: data is organised per
                        # team (search/team_data.py), so other teams never mix into Rain's.
                        from ml.teams import team_id
                        player.team_name = args.team
                        player.log('team', name=args.team, format=args.format, team_id=team_id(args.format, team['sets']))
                    agent = None
                    if args.search and not args.inspect:
                        from search.agent import RECONSTRUCTION_VERSION, SearchAgent
                        from search.config import agent_kwargs
                        brain = player.brain

                        async def preview(ctx):
                            return brain.decide(ctx, explore=False, record=False)['choice']
                        if args.preview:
                            # A gated opponent-aware team preview (docs/human-opponent.md, campaign c4);
                            # the actor's preview stays the fallback when the opponent's six is unknown.
                            from search.value_gate import make_preview
                            preview = make_preview(args.preview, preview, int(time.time()))
                            player.log('preview_policy', spec=args.preview)
                        def make_agent(over):
                            # One deployed configuration, shared with the value gate; a live A/B arm
                            # (search/ab_test.py) adds its factors' overrides.
                            sized = {k: v for k, v in (('worlds', args.search_worlds), ('engines', args.search_engines))
                                     if v is not None}
                            kwargs = agent_kwargs(**{**over, **sized})
                            made = SearchAgent(fmt=args.format, preview=preview, seed=int(time.time()), **kwargs)
                            if kwargs.get('scout'):
                                # Opponents' earlier public games (search/scout.py); gates never look online.
                                from search.scout import OnlineScout
                                cfg = kwargs['scout']
                                made.scout_source = OnlineScout(args.format, args.root / 'scout-cache',
                                                                max_games=int(cfg.get('max_games', 6)),
                                                                min_overlap=int(cfg.get('min_overlap', 4)), log=player.log)
                            return made
                        search_kwargs = agent_kwargs(worlds=args.search_worlds, engines=args.search_engines)
                        agent = make_agent({})
                        arm_over = {}

                        async def use_arm(arm, spec=None):
                            # Rebuilt only between games, when the arm's overrides change. A `team`
                            # override (campaign c13) switches the team instead of the search.
                            nonlocal agent, arm_over, team
                            want = {k: v for k, v in ((arm or {}).get('overrides') or {}).items() if k != 'team'}
                            name = arm_team(arm, spec, args.team)
                            if team is not None and name != team['name']:
                                try:
                                    team = load_team(args.teams_file, name, args.format)
                                except (KeyError, ValueError, OSError) as error:  # an unknown team plays --team, loudly
                                    player.log('ab_error', error=f'team {name!r}: {error!r}'[:300])
                                    team = load_team(args.teams_file, args.team, args.format)
                                from ml.teams import team_id
                                player.team_name = team['name']
                                player.log('arm_team', name=team['name'], team_id=team_id(args.format, team['sets']))
                            if want != arm_over:
                                old = agent
                                agent = make_agent(want)
                                agent.stats = old.stats
                                await old.close()
                                arm_over = want

                        from search.value_store import LivePriority

                        async def search_decider(ctx):
                            # Background simulation pauses while live search runs.
                            async with LivePriority():
                                return await agent.decide(ctx)
                        player.decider = search_decider
                        player.decider_trace = lambda: agent.last_trace
                        player.log('search_agent', reconstruction_version=RECONSTRUCTION_VERSION, **search_kwargs)
                    if args.inspect:
                        # Read-only transport smoke: no login, matchmaking or moves.
                        print(json.dumps(await page.evaluate("() => ({client:!!app.rooms, named:!!app.user.get('named'), title:document.title})")))
                        await page.screenshot(path=str(output / 'client.png'))
                        return
                    username, password = credentials()
                    if not username or not password:
                        raise ValueError('configured account credentials are required')
                    await player.login(username, password)
                    active = await player.active_battles()
                    if args.resume_active and not args.room and len(active) == 1:
                        args.room = active[0]
                    if active and active != [args.room]:
                        raise ValueError('existing active battle; resume its exact room before searching')
                    for game in (itertools.count() if args.continuous else range(args.games)):
                        if stop.is_set():
                            break
                        if game:
                            try:
                                await asyncio.wait_for(stop.wait(), timeout=args.game_delay)
                                break
                            except asyncio.TimeoutError:
                                pass
                        room = args.room if game == 0 else None
                        arm, ab_spec = None, None
                        if agent is not None:
                            from search import ab_test
                            try:
                                ab_spec = ab_test.load_spec()
                            except ValueError as error:  # a malformed spec plays LIVE, loudly
                                player.log('ab_error', error=repr(error)[:300])
                            if ab_spec:
                                # A resumed room keeps the arm it started with; an unknown one plays LIVE.
                                arm = ab_test.arm_for_room(ab_spec, room) if room else ab_test.next_arm(ab_spec)
                            await use_arm(arm, ab_spec)
                        if room:
                            if arm:
                                player.log('ab_arm', room=room, resumed=True,
                                           **{k: arm[k] for k in ('id', 'n', 'cell', 'random', 'overrides')})
                            await player.enter_room(room)
                        else:
                            disk = storage_status(args.root, LearningConfig.load(args.root).resource['max_disk_gb'])
                            # An explicit room resume and active finalization remain allowed.
                            if not disk['background_allowed']:
                                player.log('storage_limit', **disk)
                                print(json.dumps({'stopped': 'storage limit; active games retained', 'storage': disk}), flush=True)
                                break
                            promotion = prepare_matchmaking(player.brain, args.format)
                            player.log('matchmaking_checkpoint', **promotion)
                            if agent is not None:
                                # A gated search value is adopted only between games.
                                from search.value_store import current_value, team_pointer
                                value = current_value(team_pointer(team['name'] if team else None))
                                if value.get('sha256') != agent.value.get('sha256'):
                                    agent.set_value(value.get('path'), value.get('beta', 0.0))
                                    agent.value['sha256'] = value.get('sha256')
                                    player.log('value_checkpoint', **value)
                                # A gated screening ranker, likewise only between games.
                                from search.screen_data import current_screen, screen_pointer
                                screen = current_screen(screen_pointer(team['name'] if team else None))
                                if screen.get('sha256') != (agent.screen_model or {}).get('sha256'):
                                    agent.set_screen(screen)
                                    player.log('screen_checkpoint', **screen)
                                # A gated human opponent policy, likewise only between games.
                                from search.human.store import current_opp, opp_pointer
                                opp = current_opp(opp_pointer(team['name'] if team else None))
                                if opp.get('sha256') != (agent.opp_model or {}).get('sha256'):
                                    agent.set_opp_model(opp)
                                    player.log('opp_checkpoint', **opp)
                            room = await player.find_battle(args.format, team, args.search_timeout, stop=stop)
                            if not room:
                                if stop.is_set():
                                    break
                                raise TimeoutError('matchmaking timed out; search cancelled')
                            if arm:
                                ab_test.record_arm(ab_spec, room, arm)
                                player.log('ab_arm', room=room, **{k: arm[k] for k in ('id', 'n', 'cell', 'random', 'overrides')})
                        outcome = await player.play(room, args.format, sets=team['sets'] if team else None,
                                                    timeout=args.stall_timeout, record=not args.no_record)
                        if agent is not None:
                            save_screen_records(agent, team['name'] if team else None, room)
                        print(json.dumps({'game': game + 1, 'result': outcome}), flush=True)
                    if agent is not None:
                        player.log('search_stats', **{k: v for k, v in agent.stats.items() if k != 'errors'},
                                   errors=agent.stats['errors'][-20:])
                        await agent.close()
                    await player.close()
                    player.log('stopped', finish_active_game=True)
                    print('Browser run stopped; evidence: ' + str(output), flush=True)
                except Exception:
                    if 'player' in locals():
                        player.log('interrupted', room=locals().get('room'))
                        with __import__('contextlib').suppress(Exception):
                            await player.close()
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
    p.add_argument('--continuous', action='store_true', help='Continue until stopped or the storage admission limit is reached')
    p.add_argument('--resume-active', action='store_true', help='Resume the single active account battle after login')
    p.add_argument('--headed', action='store_true')
    p.add_argument('--chrome', action='store_true', help='Use installed Google Chrome')
    p.add_argument('--inspect', action='store_true', help='Read-only browser check; no login or games')
    p.add_argument('--no-record', action='store_true', help='Greedy inference without PPO recording')
    p.add_argument('--search', action='store_true', help='Decide turns with the simulator search agent (implies --no-record)')
    p.add_argument('--preview', default=None, help="team preview policy for --search: 'human:PATH:argmax' or 'pvalue:PATH' (value_gate.make_preview)")
    p.add_argument('--search-worlds', type=int, help='Determinized opponent worlds per decision (default: search/config.py)')
    p.add_argument('--search-engines', type=int, help='Parallel search engine processes (default: search/config.py)')
    p.add_argument('--search-timeout', type=int, default=120)
    p.add_argument('--stall-timeout', type=int, default=180)
    p.add_argument('--fallback', choices=('mcp', 'none'), default='mcp',
                   help="Transport for choices Chrome can't deliver: OpenClaw's MCP harness (default) or none")
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
        if (args.search_worlds is not None and not 1 <= args.search_worlds <= 32) or \
                (args.search_engines is not None and not 1 <= args.search_engines <= 8):
            p.error('search worlds must be 1..32 and engines 1..8')
    if not args.format.isalnum() or (args.room and not all(c.isalnum() or c == '-' for c in args.room)):
        p.error('format and room must be Showdown identifiers')
    asyncio.run(run(args))
