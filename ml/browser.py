"""Local-model decisions executed through the official browser's visible controls.

No PSClient, Player, MCP, LLM calls or direct protocol submissions. JavaScript
reads only the configured player's client state; Playwright performs all actions.
"""
from __future__ import annotations

import asyncio
from contextlib import suppress
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


# Read-only view of why move buttons may be absent; tolerates a reloading page.
CONTROLS_STATE = r"""id => {
    const r = window.app && app.rooms && app.rooms[id];
    if (!r || !r.battle) return {missing: true};
    const b = r.el.querySelector('button[name="chooseMove"]');
    return {shown: !!(b && b.offsetParent), ended: !!r.battle.ended,
        rqid: r.request ? r.request.rqid : null, waiting: !!(r.choice && r.choice.waiting),
        seeking: r.battle.seeking, atQueueEnd: !!r.battle.atQueueEnd};
}"""
# A wedged client animation queue never renders controls (it hid them for
# minutes in game 42 until the run crashed). Rejoin the room early enough to
# answer within the turn timer; give up only well after that.
CONTROLS_RELOAD_AFTER = 12
CONTROLS_GIVE_UP = 75


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
    if not side and snapshot.get('user', {}).get('name'):
        user = to_id(snapshot['user']['name'])
        side = next((parts[2] for line in snapshot.get('log', [])
                     if len(parts := line.split('|')) > 3 and parts[1] == 'player'
                     and to_id(parts[3]) == user), None)
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
    target = folder / (snapshot['room'] + '.json')
    if target.exists():
        try:
            previous = json.loads(target.read_text())
            previous.pop('updated_at', None)
            if previous == frame:
                return
        except (OSError, ValueError):
            pass
    frame['updated_at'] = time.time()
    text = json.dumps(frame)
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
        self.decider_trace = None  # optional () -> dict logged with each decided turn
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

    async def watch(self, room):
        """Observe during inference and UI waits using the existing browser only."""
        while True:
            try:
                snapshot = await self.snapshot(room)
                if self.brain and not snapshot.get('missing'):
                    target = self.brain.store.root / 'live-watch/health.json'
                    temporary = target.with_suffix('.tmp')
                    temporary.write_text(json.dumps({'room': room, 'observed_at': time.time()}))
                    temporary.replace(target)
            except Exception:
                # A failed observation must never cancel or submit a player action.
                pass
            await asyncio.sleep(.2)

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
                await self.page.wait_for_function(
                    '() => !!window.Storage?.whenTeamsLoaded?.isLoaded', timeout=15000)
                # Reuse a fully saved exact team, avoiding draft imports and
                # duplicate browser teams across service starts. These Storage
                # functions only compute the expected packed representation.
                saved_name = await self.page.evaluate("""text => {
                    const packed=Storage.packTeam(Storage.importTeam(text));
                    const teams=Storage.teams || [];
                    const saved=teams.find(t => t.team===packed && t.iconCache!=='!' &&
                        teams.filter(other => other.name===t.name).length===1);
                    return saved?.name || null;
                }""", team_text(team['sets']))
                if saved_name:
                    self.imported_team = (digest, saved_name)
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
            # Team import/selection can finish after the click handler returns.
            # Let the official client expose this exact saved payload before it
            # sends /utm; an empty team is rejected by formats requiring teams.
            await self.page.wait_for_function("""s => {
                const home=app.rooms[''];
                const team=window.Storage?.teams?.[home?.curTeamIndex];
                return home?.curFormat===s.format && team?.name===s.name &&
                    typeof team.team==='string' && team.team.length>0;
            }""", arg={'format': fmt, 'name': self.imported_team[1]}, timeout=10000)
        await home.locator('button[name="search"]').click()

    async def cancel_search(self):
        button = self.page.locator('[id="room-"] button[name="cancelSearch"]')
        if await button.count():
            await button.click()

    async def submit(self, snapshot, choice):
        await self._install_guard(snapshot)
        try:
            await self._submit(snapshot, choice)
        finally:
            await self.page.evaluate('() => {window.__psBrowserClickGuard=null}')

    async def _install_guard(self, snapshot):
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

    async def _await_controls(self, root, snapshot, *, may_reload):
        """True once move buttons render for this request; False if it is gone."""
        room = snapshot['room']
        key = (snapshot.get('request') or {}).get('rqid')
        started, reloaded = time.monotonic(), False
        while True:
            try:
                state = await self.page.evaluate(CONTROLS_STATE, room)
            except Exception:
                state = {'missing': True}  # Page is navigating.
            if state.get('shown'):
                return True
            if not state.get('missing') and (state['ended'] or state['waiting'] or
                                              (state['rqid'] is not None and state['rqid'] != key)):
                self.log('controls_request_gone', room=room, request_id=key, state=state)
                return False
            elapsed = time.monotonic() - started
            skip = root.locator('button[name="goToEnd"]:visible')
            if await skip.count():
                with suppress(Exception):
                    await skip.click(timeout=2000)
            if may_reload and not reloaded and elapsed > CONTROLS_RELOAD_AFTER:
                # Rejoining rebuilds the battle from the server and re-sends the
                # same request; no partial selection exists yet on this turn.
                reloaded = True
                self.log('controls_stalled', room=room, request_id=key, state=state)
                await self.page.goto('https://play.pokemonshowdown.com/' + room, wait_until='domcontentloaded')
                with suppress(Exception):
                    await self.page.wait_for_function(
                        'id => {const r=window.app && app.rooms && app.rooms[id];return !!(r && (r.request || (r.battle && r.battle.ended)));}',
                        arg=room, timeout=30000)
                with suppress(Exception):
                    await self._install_guard(snapshot)
                continue
            if elapsed > CONTROLS_GIVE_UP:
                if self.log_path:
                    with suppress(Exception):
                        await self.page.screenshot(path=str(self.log_path.parent / f'controls-stall-{key}.png'))
                self.log('controls_never_shown', room=room, request_id=key, state=state)
                raise TimeoutError('move controls never rendered; inspect before resuming')
            await asyncio.sleep(.25)

    async def _submit(self, snapshot, choice):
        room = snapshot['room']
        key = (snapshot.get('request') or {}).get('rqid')
        if choice not in legal_choices(snapshot.get('request')):
            raise ValueError('model choice is outside the legal request mask')
        current = await self.snapshot(room)
        if (current.get('request') or {}).get('rqid') != key or current.get('waiting') or current.get('partial'):
            raise ValueError('request changed or browser already has a choice')
        root = self.page.locator(f'[id="room-{room}"]')
        if not choice.startswith('team '):
            # Use the official playback control to expose the current request
            # immediately; long move animations must not consume the game timer.
            # Optional: the button can detach mid-animation; the controls wait
            # below handles a client that is still replaying.
            skip = root.locator('button[name="goToEnd"]:visible')
            if await skip.count():
                with suppress(Exception):
                    await skip.click(timeout=5000)

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
            clicked = False
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
                    clicked = True
                    select = root.locator('button[name="selectSwitch"]:visible')
                    if await select.count():
                        await select.click()
                    await click('chooseSwitch', int(tokens[1]) - 1)
                    target = root.locator(f'button[name="chooseSwitchTarget"][value="{slot}"]:visible')
                    if await target.count():
                        await target.click()
                else:
                    events = {'mega': 'megaevo', 'megax': 'megaevox', 'megay': 'megaevoy', 'terastallize': 'terastallize'}
                    # Controls (and their evolution checkboxes) appear only after
                    # the turn's animations; wait for this slot's move buttons.
                    # Only an untouched turn may rejoin the room to unstick them.
                    if not await self._await_controls(root, snapshot, may_reload=not clicked):
                        self.log('submit_abandoned', room=room, request_id=key, choice=choice)
                        return
                    for flag, name in events.items():
                        checkbox = root.locator(f'input[name="{name}"]:visible')
                        if await checkbox.count():
                            await checkbox.set_checked(flag in tokens)
                        elif flag in tokens:
                            raise ValueError('required evolution control is missing')
                    move = int(tokens[1])
                    # Official client (oldclient/client-battle.js): every enabled
                    # listed move is button i+1, including a listed Struggle; the
                    # synthetic Struggle button value 0 exists only when all listed
                    # moves are disabled.
                    if not any(not m.get('disabled') for m in snapshot['request']['active'][slot]['moves']):
                        move = 0
                    clicked = True
                    await click('chooseMove', move)
                    target = next((t for t in tokens[2:] if t.lstrip('-').isdigit()), None)
                    if target is not None:
                        await click('chooseMoveTarget', int(target))
        # An incomplete click sequence never counts as a submitted action.
        await self.page.wait_for_function("s => {const r=app.rooms[s.room];return r && ((r.choice && r.choice.waiting) || r.battle.ended || (r.request && r.request.rqid !== s.rqid));}", arg={'room': room, 'rqid': key}, timeout=10000)
        self.log('submitted', room=room, request_id=key, choice=choice)

    async def play(self, room, fmt, *, sets=None, timeout=180, record=True):
        observer = asyncio.create_task(self.watch(room))
        try:
            return await self._play(room, fmt, sets=sets, timeout=timeout, record=record)
        finally:
            observer.cancel()
            with suppress(asyncio.CancelledError):
                await observer

    async def _play(self, room, fmt, *, sets=None, timeout=180, record=True):
        if self.brain is None:
            raise ValueError('a local model is required')
        last_progress = time.monotonic()
        pinned = None
        previous = None
        retries = {}
        sheets_attempted = False
        while True:
            snapshot = await self.snapshot(room)
            if snapshot.get('missing'):
                raise ValueError('battle room disappeared; rejoin before resuming')
            outcome = result(snapshot)
            if not outcome.get('ongoing'):
                side = watch_frame(snapshot)['side'] or self.brain.room_side(room)
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
                self.log('terminal', result=outcome, side=side, experience=saved, log=snapshot['log'])
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
                if not sheets_attempted:
                    offer = self.page.locator(f'[id="room-{room}"] button[name="send"][value="/acceptopenteamsheets"]:visible')
                    if await offer.count():
                        sheets_attempted = True
                        try:
                            await offer.click(timeout=1000)
                        except Exception as error:
                            # Playwright is optional for protocol/offline users.
                            # Only a disappearing optional offer may time out;
                            # connection failures still stop the owner.
                            if not (isinstance(error, TimeoutError) or
                                    (type(error).__name__ == 'TimeoutError' and
                                     type(error).__module__.startswith('playwright.'))):
                                raise
                            self.log('team_sheets_offer_expired', room=room)
                        else:
                            self.last_submission = time.monotonic()
                            self.log('team_sheets_accepted', room=room)
                        continue  # Re-read the request after the UI may have changed.
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
                    trace = getattr(self, 'decider_trace', None)
                    if trace is not None:
                        decision['search'] = trace()  # top values, margin, worlds, sims: value-learning evidence
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
