"""Local-model decisions executed through the official browser's visible controls.

No PSClient, Player, MCP, LLM calls or direct protocol submissions. JavaScript
reads only the configured player's client state; Playwright performs all actions.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from battle_state import legal_choices, observe, to_id
from ml.storage import fingerprint
from ml.teams import team_id

SNAPSHOT = r"""(id) => {
    if (!window.app || !app.rooms) throw new Error('Unsupported Showdown client');
    const room = app.rooms[id];
    const user = {name: app.user.get('name'), named: !!app.user.get('named')};
    if (!room || !room.battle) return {room: id, user, missing: true};
    const request = room.request ? JSON.parse(JSON.stringify(room.request)) : null;
    return {room: id, user, request, log: room.battle.stepQueue.slice(),
        ended: !!room.battle.ended, waiting: !!(room.choice && room.choice.waiting),
        partial: !!(room.choice && !room.choice.waiting &&
            (room.choice.done || (room.choice.choices || []).some(c => c && c !== 'pass'))),
        previewOrder: room.choice && room.choice.teamPreview,
        error: [...room.el.querySelectorAll('.error')].map(e => e.textContent).join('\n')};
}"""


def request_key(snapshot):
    request = snapshot.get('request') or {}
    return (request.get('rqid'), fingerprint(request))


def result(snapshot):
    for line in reversed(snapshot.get('log', [])):
        if line.startswith('|win|'):
            return {'room': snapshot['room'], 'winner': line.split('|', 2)[2]}
        if line in ('|tie', '|tie|'):
            return {'room': snapshot['room'], 'tie': True}
    return {'room': snapshot['room'], 'ongoing': True}


def watch_frame(snapshot):
    """Export battle protocol only; never requests, chat, or player identities."""
    from ml.scout import PUBLIC_KINDS
    side = (snapshot.get('request') or {}).get('side', {}).get('id') or snapshot.get('side')
    names = {s: 'Your agent' if s == side else 'Opponent' for s in ('p1', 'p2')}
    players = {}
    log = []
    visible = PUBLIC_KINDS | {'teampreview', 'teamsize', 'upkeep', 'inactive', 'inactiveoff', 'cant', 'swap', 'start'}
    for line in snapshot.get('log', []):
        parts = line.split('|')
        if len(parts) < 2 or not (parts[1] in visible or parts[1].startswith('-')):
            continue
        if parts[1] == 'player' and len(parts) > 3:
            players[parts[3]] = parts[2]
            avatar = parts[4] if len(parts) > 4 and parts[4].isdigit() else '1'
            line = f"|player|{parts[2]}|{names.get(parts[2], 'Player')}|{avatar}"
        elif parts[1] == 'win' and len(parts) > 2:
            line = '|win|' + names.get(players.get(parts[2]), 'Winner')
        log.append(line)
    return {'room': snapshot['room'], 'side': side, 'log': log,
            'turn': next((int(line.split('|')[2]) for line in reversed(log)
                          if line.startswith('|turn|') and line.split('|')[2].isdigit()), 0),
            'live': bool(result(snapshot).get('ongoing')), 'source': 'browser-player-relay'}


def publish_watch(root, snapshot):
    folder = Path(root) / 'live-watch'
    folder.mkdir(exist_ok=True)
    frame = watch_frame(snapshot)
    text = json.dumps(frame)
    target = folder / (snapshot['room'] + '.json')
    if target.exists() and target.read_text() == text:
        return
    for target in (target, folder / 'current.json'):
        temporary = target.with_suffix('.tmp')
        temporary.write_text(text)
        temporary.replace(target)


def context(snapshot, fmt, sets=None):
    room, request = snapshot['room'], snapshot.get('request')
    if not room.startswith('battle-' + fmt + '-'):
        raise ValueError('format does not match battle room')
    if snapshot.get('waiting') or not result(snapshot).get('ongoing'):
        raise ValueError('no actionable request')
    choices = legal_choices(request)
    if not choices:
        raise ValueError('no actionable request')
    state = observe(request, snapshot['log'], snapshot['user']['name'], room)
    state['choices'] = choices
    return {'room': room, 'format': fmt, 'source': 'ladder',
            'team_id': team_id(fmt, sets) if sets else '', 'team_sets': sets,
            'turn': state['turn'], 'request': request, 'choices': choices,
            'state': state, 'public_log': snapshot['log']}


def team_text(sets):
    """Own stored sets rendered as the client's standard text import format."""
    labels = {'hp': 'HP', 'atk': 'Atk', 'def': 'Def', 'spa': 'SpA', 'spd': 'SpD', 'spe': 'Spe'}
    blocks = []
    for mon in sets:
        title = mon['species']
        if mon.get('name') and mon['name'] != mon['species']:
            title = f"{mon['name']} ({title})"
        if mon.get('gender') in ('M', 'F'):
            title += ' (' + mon['gender'] + ')'
        if mon.get('item'):
            title += ' @ ' + mon['item']
        lines = [title]
        for key, label in [('ability', 'Ability'), ('level', 'Level'), ('teraType', 'Tera Type'), ('happiness', 'Happiness')]:
            if key in mon:
                lines.append(f'{label}: {mon[key]}')
        if mon.get('shiny'):
            lines.append('Shiny: Yes')
        for key, label in [('evs', 'EVs'), ('ivs', 'IVs')]:
            values = mon.get(key, {})
            if values:
                lines.append(label + ': ' + ' / '.join(f'{values[k]} {v}' for k, v in labels.items() if k in values))
        if mon.get('nature'):
            lines.append(mon['nature'] + ' Nature')
        lines.extend('- ' + move for move in mon['moves'])
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks)


class BrowserPlayer:
    def __init__(self, page, brain=None, *, log_path=None, decision_interval=1.0):
        if decision_interval < 0:
            raise ValueError('decision interval must be nonnegative')
        self.page, self.brain = page, brain
        # Optional async ctx -> choice override (search agent); never recorded.
        self.decider = None
        self.decision_interval = decision_interval
        self.last_submission = 0.0
        self.answered = set()
        self.imported_team = None
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch(mode=0o600, exist_ok=True)
            self.log_path.chmod(0o600)

    def log(self, kind, **data):
        if self.log_path:
            with self.log_path.open('a') as handle:
                handle.write(json.dumps({'time': time.time(), 'kind': kind, **data}) + '\n')

    async def snapshot(self, room):
        snapshot = await self.page.evaluate(SNAPSHOT, room)
        if self.brain and not snapshot.get('missing'):
            publish_watch(self.brain.store.root, snapshot)
        return snapshot

    async def login(self, username, password):
        # A persistent profile re-authenticates on its own a few seconds after
        # load; let that finish instead of racing it with the login form.
        try:
            await self.page.wait_for_function("() => app.user.get('named')", timeout=10000)
        except Exception:
            pass
        identity = await self.page.evaluate("() => ({name:app.user.get('name'),named:app.user.get('named')})")
        if identity['named']:
            if to_id(identity['name']) != to_id(username):
                raise ValueError('browser is logged in as a different account')
            return
        await self.page.locator('button[name="login"]:visible').first.click()
        await self.page.locator('.ps-popup input[name="username"]:not([readonly])').fill(username)
        await self.page.locator('.ps-popup button[type="submit"]').click()
        # Require a registered-account password form; never create a guest identity.
        field = self.page.locator('.ps-popup input[name="password"]')
        await field.wait_for(timeout=15000)
        try:
            await field.fill(password)
            await self.page.locator('.ps-popup button[type="submit"]').click()
            await self.page.wait_for_function("name => app.user.get('named') && app.user.get('userid') === name", arg=to_id(username))
        except Exception:
            # Playwright action traces can contain filled values. Do not expose
            # a credential-bearing exception through the CLI or event logs.
            raise ValueError('browser account login failed') from None

    async def search(self, fmt, team=None):
        await self.page.locator('a[href="/"]:visible').first.click()
        home = self.page.locator('[id="room-"]')
        await home.locator('button[name="format"]').click()
        search = self.page.locator('.ps-popup input[name="search"]')
        await search.fill(fmt)
        await search.press('End')  # Classic client filters formats on keyup.
        await self.page.locator(f'button[name="selectFormat"][value="{fmt}"]').click()
        if team:
            # Import via the actual teambuilder UI, then select that exact team.
            digest = fingerprint(team)
            if not self.imported_team or self.imported_team[0] != digest:
                name = 'Browser ' + uuid.uuid4().hex[:12]
                await home.locator('button[name="joinRoom"][value="teambuilder"]').click()
                builder = self.page.locator('[id="room-teambuilder"]')
                await builder.locator('button[name="newTop"][value="team"]').click()
                await builder.locator('input.teamnameedit').fill(name)
                await builder.locator('input.teamnameedit').press('Tab')
                await builder.locator('button[name="import"]').first.click()
                await builder.locator('textarea').fill(team_text(team['sets']))
                await builder.locator('button[name="saveImport"].savebutton').click()
                await self.page.locator('a[href="/"]:visible').first.click()
                self.imported_team = (digest, name)
            await home.locator('button[name="team"]').click()
            # The popup exposes unformatted teams as well as format-specific ones.
            selected = self.page.locator('button[name="selectTeam"]').filter(has_text=self.imported_team[1])
            if not await selected.count():
                await self.page.locator('button[name="moreTeams"]').click()
            await selected.click()
        await home.locator('button[name="search"]').click()

    async def cancel_search(self):
        button = self.page.locator('[id="room-"] button[name="cancelSearch"]')
        if await button.count():
            await button.click()

    async def submit(self, snapshot, choice):
        # A locator can wait while a newer request arrives. Cancel the click at
        # DOM dispatch time as well as checking freshness before selecting it.
        await self.page.evaluate(r"""s => {
            window.__psBrowserClickGuard = {room:s.room, rqid:(s.request||{}).rqid, blocked:false};
            if (window.__psBrowserGuardInstalled) return;
            window.__psBrowserGuardInstalled = true;
            document.addEventListener('click', e => {
                const g=window.__psBrowserClickGuard;
                if (!g) return;
                const r=app.rooms[g.room];
                if (!r || !r.el.contains(e.target)) return;
                // The client annotates its request copy while rendering; only the
                // server request id identifies a newer request.
                if (!r.request || r.request.rqid!==g.rqid || (r.choice && r.choice.waiting)) {
                    g.blocked=true; e.preventDefault(); e.stopImmediatePropagation();
                }
            }, true);
        }""", {'room': snapshot['room'], 'request': snapshot['request']})
        try:
            await self._submit(snapshot, choice)
        finally:
            await self.page.evaluate('() => {window.__psBrowserClickGuard=null}')

    async def _submit(self, snapshot, choice):
        room = snapshot['room']
        key = (snapshot.get('request') or {}).get('rqid')
        if choice not in legal_choices(snapshot.get('request')):
            raise ValueError('model choice is outside the legal request mask')
        current = await self.snapshot(room)
        if (current.get('request') or {}).get('rqid') != key or current.get('waiting') or current.get('partial'):
            raise ValueError('request changed or browser already has a choice')
        root = self.page.locator(f'[id="room-{room}"]')

        async def click(name, value):
            await root.locator(f'button[name="{name}"][value="{value}"]').click(timeout=90000)
            if await self.page.evaluate('() => !!window.__psBrowserClickGuard?.blocked'):
                raise ValueError('request changed while waiting for browser control')

        if choice.startswith('team '):
            for number in map(int, choice[5:].split(',')):
                current = await self.snapshot(room)
                if (current.get('request') or {}).get('rqid') != key or current.get('waiting'):
                    raise ValueError('request changed during preview clicks')
                # The client builds its preview choice when controls render; a fast
                # decision can arrive first, so wait for the order to exist.
                for _ in range(40):
                    if current.get('previewOrder'):
                        break
                    await asyncio.sleep(.25)
                    current = await self.snapshot(room)
                    if (current.get('request') or {}).get('rqid') != key or current.get('waiting'):
                        raise ValueError('request changed during preview clicks')
                if not current.get('previewOrder'):
                    raise ValueError('team preview controls never rendered')
                # Buttons index the dynamically reordered preview, not the party.
                position = current['previewOrder'].index(number)
                await click('chooseTeamPreview', position)
        else:
            for slot, command in enumerate(choice.split(', ')):
                tokens = command.split()
                if tokens == ['pass']:
                    continue  # The official client fills inactive/unused slots.
                if tokens[0] == 'default':
                    raise ValueError('default has no explicit browser control; stop for inspection')
                current = await self.snapshot(room)
                if (current.get('request') or {}).get('rqid') != key or current.get('waiting'):
                    raise ValueError('request changed during action clicks')
                if tokens[0] == 'switch':
                    select = root.locator('button[name="selectSwitch"]:visible')
                    if await select.count():
                        await select.click()
                    await click('chooseSwitch', int(tokens[1]) - 1)
                    target = root.locator(f'button[name="chooseSwitchTarget"][value="{slot}"]:visible')
                    if await target.count():
                        await target.click()
                else:
                    events = {'mega': 'megaevo', 'megax': 'megaevox', 'megay': 'megaevoy', 'terastallize': 'terastallize'}
                    for flag, name in events.items():
                        checkbox = root.locator(f'input[name="{name}"]:visible')
                        if await checkbox.count():
                            await checkbox.set_checked(flag in tokens)
                        elif flag in tokens:
                            raise ValueError('required evolution control is missing')
                    move = int(tokens[1])
                    # The UI labels Struggle with zero; requests expose move one.
                    if to_id(snapshot['request']['active'][slot]['moves'][move - 1].get('id', '')) == 'struggle':
                        move = 0
                    await click('chooseMove', move)
                    target = next((t for t in tokens[2:] if t.lstrip('-').isdigit()), None)
                    if target is not None:
                        await click('chooseMoveTarget', int(target))
        # An incomplete click sequence never counts as a submitted action.
        await self.page.wait_for_function("s => {const r=app.rooms[s.room];return r && ((r.choice && r.choice.waiting) || r.battle.ended || (r.request && r.request.rqid !== s.rqid));}", arg={'room': room, 'rqid': key}, timeout=10000)
        self.log('submitted', room=room, request_id=key, choice=choice)

    async def play(self, room, fmt, *, sets=None, timeout=180, record=True):
        if self.brain is None:
            raise ValueError('a local model is required')
        last_progress = time.monotonic()
        pinned = None
        previous = None
        retries = {}
        sheets_accepted = False
        while True:
            snapshot = await self.snapshot(room)
            if snapshot.get('missing'):
                raise ValueError('battle room disappeared; rejoin before resuming')
            outcome = result(snapshot)
            if not outcome.get('ongoing'):
                side = (snapshot.get('request') or {}).get('side', {}).get('id') or self.brain.room_side(room)
                pending = self.brain.resume(room, side)
                if pending and pending['steps'] and pending['steps'][-1].get('submitted') is False:
                    self.brain.reject(room, side, reason='browser-unconfirmed-terminal-proposal')
                user = next((line.split('|')[3] for line in snapshot['log'] if line.startswith('|player|' + side + '|')), snapshot['user']['name'])
                saved = self.brain.finish(room, outcome, user, side, public_log=snapshot['log'])
                if self.decider is not None:
                    # Search games are not PPO episodes, but their public log still
                    # feeds the store (scout samples, opponent sets, learning data).
                    try:
                        from ml.scout import ingest_public
                        saved = {**(saved or {}), 'public_experience': ingest_public(
                            self.brain.store, fmt, snapshot['log'], 'own-live-game', 'own-' + room,
                            features=self.brain.features)}
                    except Exception as error:
                        saved = {**(saved or {}), 'public_experience': {'imported': False, 'error': type(error).__name__}}
                self.log('terminal', result=outcome, experience=saved, log=snapshot['log'])
                return outcome
            key = request_key(snapshot)
            rejected = previous and (any(line.startswith('|error|') for line in snapshot['log'][previous['log_length']:]) or
                (not snapshot.get('waiting') and key[0] == previous['key'][0] and key != previous['key']))
            if rejected:
                retries[key[0]] = retries.get(key[0], 0) + 1
                self.brain.reject(room, previous['side'], reason='browser-server-rejection')
                self.answered.discard(previous['key'])
                previous = None
                if retries[key[0]] >= 3:
                    raise ValueError('repeated server rejection; inspect this request before resuming')
            if not snapshot.get('waiting') and legal_choices(snapshot.get('request')) and key not in self.answered:
                if not sheets_accepted:
                    offer = self.page.locator(f'[id="room-{room}"] button[name="send"][value="/acceptopenteamsheets"]:visible')
                    if await offer.count():
                        await offer.click()
                        sheets_accepted = True
                        self.last_submission = time.monotonic()
                        self.log('team_sheets_accepted', room=room)
                        continue
                delay = self.last_submission + self.decision_interval - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                    continue  # Read a fresh request after the pacing delay.
                if snapshot.get('partial'):
                    raise ValueError('partially selected browser action; inspect before resuming')
                ctx = context(snapshot, fmt, sets)
                model = self.brain.model(fmt)
                identity = (model.revision, model.checkpoint_sha256, model.feature_profile,
                            model.policy_temperature, model.preview_temperature)
                if pinned and pinned != identity:
                    raise ValueError('checkpoint changed during browser game')
                pinned = identity
                side = ctx['request']['side']['id']
                episode = self.brain.resume(room, side)
                if episode:
                    model = self.brain.model(fmt)
                    policy = episode['collecting_policy']
                    if (episode['team'] != ctx['team_id'] or episode['format'] != fmt or
                        episode['revision'] != model.revision or policy['checkpoint_sha256'] != model.checkpoint_sha256 or
                        policy['feature_profile'] != model.feature_profile or
                        policy['policy_temperature'] != model.policy_temperature or
                        policy['preview_temperature'] != model.preview_temperature or not record):
                        raise ValueError('pending recording does not match team/checkpoint/mode')
                saved = episode['steps'][-1] if episode and episode['steps'] else None
                if saved and saved['request_id'] == ctx['request'].get('rqid', fingerprint(ctx['request'])):
                    model = self.brain.model(fmt)
                    if episode['team'] != ctx['team_id'] or episode['revision'] != model.revision or episode['collecting_policy']['checkpoint_sha256'] != model.checkpoint_sha256:
                        raise ValueError('pending recording does not match team/checkpoint')
                    if saved.get('submitted'):
                        raise ValueError('submission acknowledgement ambiguous; inspect existing room')
                    if fingerprint(saved.get('snapshot', {}).get('request')) != fingerprint(ctx['request']):
                        raise ValueError('recovered request differs from the recorded proposal; inspect before resuming')
                    decision = {'choice': saved['choice'], 'recorded': True, 'recovered': True}
                elif self.decider is not None:
                    if record:
                        raise ValueError('search decisions are not PPO recordings; use --no-record')
                    started = time.perf_counter()
                    decision = {'choice': await self.decider(ctx), 'revision': pinned[0], 'recorded': False,
                                'decider': getattr(self.decider, '__name__', 'search'),
                                'inference_ms': round((time.perf_counter() - started) * 1000, 1)}
                else:
                    decision = self.brain.decide(ctx, explore=record, record=record)
                if not decision.get('recovered') and decision['revision'] != pinned[0]:
                    raise ValueError('checkpoint changed during model inference')
                self.log('decision', room=room, request=ctx['request'], state=ctx['state'], decision=decision)
                if self.decider is not None:
                    # Search thinks for seconds; the client may rewrite its copy of
                    # the same server request meanwhile. Submit against a fresh
                    # snapshot of that same request id, never a newer request.
                    fresh = await self.snapshot(room)
                    if ((fresh.get('request') or {}).get('rqid') != (snapshot.get('request') or {}).get('rqid')
                            or fresh.get('waiting') or decision['choice'] not in legal_choices(fresh.get('request'))):
                        raise ValueError('request changed during search; inspect before resuming')
                    snapshot = fresh
                await self.submit(snapshot, decision['choice'])
                self.last_submission = time.monotonic()
                if decision.get('recorded'):
                    episode = self.brain.pending[(room, side)]
                    episode['steps'][-1]['submitted'] = True
                    self.brain.store.save_episode(episode)
                self.answered.add(key)
                previous = {'key': key, 'side': side, 'log_length': len(snapshot['log'])}
                last_progress = time.monotonic()
            if time.monotonic() - last_progress > timeout:
                self.log('stalled', room=room)
                raise TimeoutError('battle stalled; recording retained, no outcome assigned')
            await asyncio.sleep(.25)
