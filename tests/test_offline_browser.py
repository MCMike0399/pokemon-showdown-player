"""Exercise decision/transport boundaries without accounts or network calls."""
import asyncio
import time
from copy import deepcopy

import pytest

import ml.browser as browser
from ml.browser import BrowserPlayer, CONTROLS_STATE, SNAPSHOT, context, result, team_text

FMT = 'gen9championsvgc2026regmc'
ROOM = 'battle-' + FMT + '-123'


def snapshot(preview=False, forced=False):
    party = [{'ident': 'p2: ' + species, 'details': species + ', L50',
              'condition': '100/100', 'active': i < 2}
             for i, species in enumerate(['Politoed', 'Garchomp', 'Golisopod', 'Farigiraf', 'Archaludon', 'Rillaboom'])]
    request = {'rqid': 4, 'side': {'id': 'p2', 'pokemon': party}}
    if preview:
        request.update(teamPreview=True, maxChosenTeamSize=4)
    elif forced:
        request['forceSwitch'] = [True, True]
        party[0]['condition'] = party[1]['condition'] = '0 fnt'
        for mon in party[3:]:
            mon['condition'] = '0 fnt'
    else:
        request['active'] = [{'canMegaEvo': True, 'moves': [{'id': 'tackle', 'move': 'Tackle', 'target': 'normal', 'pp': 10}]},
                             {'moves': [{'id': 'protect', 'move': 'Protect', 'target': 'self', 'pp': 10}]}]
    return {'room': ROOM, 'user': {'name': 'Configured', 'named': True}, 'request': request,
            'log': ['|player|p1|Opponent', '|player|p2|Configured', '|turn|3'],
            'waiting': False, 'partial': False, 'previewOrder': [1, 2, 3, 4, 5, 6]}


class Locator:
    def __init__(self, page, selector):
        self.page, self.selector = page, selector

    def locator(self, selector):
        return Locator(self.page, selector)

    async def count(self):
        return int('megaevo' in self.selector or 'chooseSwitchTarget' in self.selector)

    async def click(self, **kwargs):
        self.page.clicked.append(self.selector)
        if 'chooseTeamPreview' in self.selector:
            pos = int(self.selector.split('value="')[1].split('"')[0])
            order = self.page.state['previewOrder']
            done = self.page.done
            order[done], order[pos] = order[pos], order[done]
            self.page.done += 1

    async def set_checked(self, value):
        self.page.checked.append((self.selector, value))


class Page:
    def __init__(self, state):
        self.state, self.clicked, self.checked, self.done = state, [], [], 0

    controls = {'shown': True}

    async def evaluate(self, script, room=None):
        if script == CONTROLS_STATE:
            return dict(self.controls)
        return deepcopy(self.state) if script == SNAPSHOT else None

    def locator(self, selector):
        return Locator(self, selector)

    async def wait_for_function(self, *args, **kwargs):
        self.state['waiting'] = True


def test_preview_reorders_buttons_by_current_party_position():
    page = Page(snapshot(preview=True))
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'team 4,1,6,2'))
    assert page.state['previewOrder'][:4] == [4, 1, 6, 2]
    assert page.clicked == ['button[name="chooseTeamPreview"][value="3"]',
                            'button[name="chooseTeamPreview"][value="3"]',
                            'button[name="chooseTeamPreview"][value="5"]',
                            'button[name="chooseTeamPreview"][value="3"]']


def test_joint_target_and_mega_are_clicked_without_chat_commands():
    page = Page(snapshot())
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'move 1 2 mega, move 1'))
    assert page.clicked == ['button[name="chooseMove"][value="1"]',
                            'button[name="chooseMoveTarget"][value="2"]',
                            'button[name="chooseMove"][value="1"]']
    assert page.checked[0][1] is True and page.checked[-1][1] is False


def test_scarce_forced_switch_selects_correct_slot_and_leaves_pass_to_client():
    page = Page(snapshot(forced=True))
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'pass, switch 3'))
    assert page.clicked == ['button[name="chooseSwitch"][value="2"]',
                            'button[name="chooseSwitchTarget"][value="1"]:visible']


@pytest.mark.parametrize('change', ['stale', 'waiting', 'partial', 'illegal'])
def test_refuses_stale_duplicate_partial_and_illegal_choices(change):
    state = snapshot()
    page = Page(deepcopy(state))
    choice = 'move 1 2, move 1'
    if change == 'stale':
        page.state['request']['rqid'] += 1
    elif change == 'illegal':
        choice = 'move 99, move 1'
    else:
        page.state[change] = True
    with pytest.raises(ValueError):
        asyncio.run(BrowserPlayer(page).submit(state, choice))
    assert not page.clicked


def test_context_uses_private_player_perspective_and_terminal_is_required():
    state = snapshot()
    ctx = context(state, FMT)
    assert ctx['state']['side_id'] == 'p2' and ctx['turn'] == 3
    assert result(state)['ongoing']
    state['log'].append('|win|Configured')
    assert result(state)['winner'] == 'Configured'
    with pytest.raises(ValueError):
        context(state, FMT)
    with pytest.raises(ValueError, match='format'):
        context(snapshot(), 'gen9randombattle')


def test_import_retains_nature_and_stat_points_without_inventing_ivs():
    text = team_text([{'species': 'Politoed', 'ability': 'Drizzle', 'moves': ['Protect'],
                       'nature': 'Bold', 'evs': {'hp': 32, 'def': 32}, 'level': 50}])
    assert '32 HP / 32 Def' in text and 'Bold Nature' in text and 'Level: 50' in text
    assert 'IVs' not in text


def test_local_model_loop_records_exactly_one_sample_and_terminal(tmp_path):
    from ml.brain import Brain
    from ml.features import Features
    from ml.storage import Store
    store = Store(tmp_path)
    try:
        brain = Brain(store, Features())
        page = Page(snapshot())
        player = BrowserPlayer(page, brain, log_path=tmp_path / 'events.jsonl')
        submitted = []
        async def submit(state, choice):
            submitted.append(choice)
            page.state['log'].append('|win|Configured')
        player.submit = submit
        outcome = asyncio.run(player.play(ROOM, FMT))
        assert outcome['winner'] == 'Configured' and len(submitted) == 1
        row = store.room_episodes(ROOM, 'p2')[0]
        assert row['status'] == 'complete' and row['outcome'] == 1
        assert row['on_policy'] and len(row['steps']) == 1
        assert row['steps'][0]['submitted'] is True
        assert row['steps'][0]['choice'] == submitted[0]
        assert not store.db.execute('SELECT id FROM jobs').fetchall()
    finally:
        store.close()


def test_pacing_reads_new_request_before_model_decision(tmp_path):
    from ml.brain import Brain
    from ml.features import Features
    from ml.storage import Store
    store = Store(tmp_path)
    try:
        page = Page(snapshot())
        player = BrowserPlayer(page, Brain(store, Features()), decision_interval=.05)
        player.last_submission = time.monotonic()
        submitted = []
        async def submit(state, choice):
            submitted.append(state['request']['rqid'])
            page.state['log'].append('|win|Configured')
        player.submit = submit
        async def run():
            async def update():
                await asyncio.sleep(.01)
                page.state['request']['rqid'] = 5
            task = asyncio.create_task(update())
            await player.play(ROOM, FMT)
            await task
        asyncio.run(run())
        assert submitted == [5]
        assert store.room_episodes(ROOM, 'p2')[0]['steps'][0]['request_id'] == 5
    finally:
        store.close()


def test_recovery_refuses_changed_request_with_same_rqid(tmp_path):
    from ml.brain import Brain
    from ml.features import Features
    from ml.storage import Store
    store = Store(tmp_path)
    try:
        brain = Brain(store, Features())
        original = snapshot()
        brain.decide(context(original, FMT), explore=True, record=True)
        changed = deepcopy(original)
        changed['request']['side']['pokemon'][0]['condition'] = '20/100'
        player = BrowserPlayer(Page(changed), brain)
        with pytest.raises(ValueError, match='differs from the recorded proposal'):
            asyncio.run(player.play(ROOM, FMT))
        row = store.room_episodes(ROOM, 'p2')[0]
        assert row['status'] == 'pending' and row['steps'][0]['submitted'] is False
        assert row['steps'][0]['snapshot']['request'] == original['request']
    finally:
        store.close()


def test_disappearing_optional_sheet_offer_does_not_stop_the_battle(tmp_path):
    from ml.brain import Brain
    from ml.features import Features
    from ml.storage import Store
    class Offer:
        calls = 0
        async def count(self):
            return 1
        async def click(self, **kwargs):
            self.calls += 1
            raise TimeoutError('optional offer detached before dispatch')
    offer = Offer()
    class OfferPage(Page):
        def locator(self, selector):
            if '/acceptopenteamsheets' in selector:
                return offer
            return super().locator(selector)
    store = Store(tmp_path)
    try:
        page = OfferPage(snapshot())
        player = BrowserPlayer(page, Brain(store, Features()), log_path=tmp_path / 'events.jsonl')
        submissions = []
        async def submit(state, choice):
            submissions.append(choice)
            page.state['log'].append('|win|Configured')
        player.submit = submit
        outcome = asyncio.run(player.play(ROOM, FMT))
        assert outcome['winner'] == 'Configured'
        assert len(submissions) == 1 and offer.calls == 1
    finally:
        store.close()


def test_search_waits_for_async_team_selection_before_clicking_battle():
    from ml.storage import fingerprint
    team = {'format': FMT, 'sets': [{'species': 'Politoed', 'moves': ['Protect']}]}
    class SearchPage:
        ready = False
        searched = False
        class SearchLocator:
            def __init__(self, page, selector):
                self.page, self.selector = page, selector
            @property
            def first(self):
                return self
            def locator(self, selector):
                return self.__class__(self.page, selector)
            def filter(self, **kwargs):
                return self
            async def count(self):
                return 1
            async def fill(self, value):
                pass
            async def press(self, value):
                pass
            async def click(self, **kwargs):
                if 'selectTeam' in self.selector:
                    asyncio.get_running_loop().call_soon(setattr, self.page, 'ready', True)
                elif self.selector == 'button[name="search"]':
                    assert self.page.ready, 'Battle clicked before selected team became available'
                    self.page.searched = True
        def locator(self, selector):
            return self.SearchLocator(self, selector)
        async def wait_for_function(self, *args, **kwargs):
            while not self.ready:
                await asyncio.sleep(0)
    page = SearchPage()
    player = BrowserPlayer(page)
    player.imported_team = (fingerprint(team), 'Selected team')
    asyncio.run(player.search(FMT, team))
    assert page.searched


def test_wedged_move_controls_rejoin_the_room_then_click(monkeypatch):
    monkeypatch.setattr(browser, 'CONTROLS_RELOAD_AFTER', 0)
    page = Page(snapshot())
    page.controls = {'shown': False, 'ended': False, 'waiting': False, 'rqid': 4, 'seeking': float('inf')}
    page.visited = []

    async def goto(url, **kwargs):
        page.visited.append(url)
        page.controls = {'shown': True}

    async def wait_for_function(script, arg=None, **kwargs):
        if not isinstance(arg, str):  # Rejoin waits on the room id; submit on the request.
            page.state['waiting'] = True
    page.goto, page.wait_for_function = goto, wait_for_function
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'move 1 2, move 1'))
    assert page.visited == ['https://play.pokemonshowdown.com/' + ROOM]
    assert page.clicked[-3:] == ['button[name="chooseMove"][value="1"]',
                                 'button[name="chooseMoveTarget"][value="2"]',
                                 'button[name="chooseMove"][value="1"]']


def test_battle_ending_while_controls_are_hidden_submits_nothing():
    page = Page(snapshot())
    page.controls = {'shown': False, 'ended': True, 'waiting': False, 'rqid': 4}
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'move 1 1, move 1'))
    assert not any('chooseMove' in selector for selector in page.clicked)


def test_detaching_skip_button_does_not_abort_the_turn():
    # Live 2026-10-09: goToEnd resolved, then detached mid-animation; the
    # unguarded click timed out and crashed the runner during a game.
    page = Page(snapshot())
    original = Locator.count, Locator.click

    async def count(self):
        return 1 if 'goToEnd' in self.selector else await original[0](self)

    async def click(self, **kwargs):
        if 'goToEnd' in self.selector:
            raise TimeoutError('Locator.click: element was detached from the DOM')
        return await original[1](self, **kwargs)
    Locator.count, Locator.click = count, click
    try:
        asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'move 1 2, move 1'))
    finally:
        Locator.count, Locator.click = original
    assert page.clicked[-3:] == ['button[name="chooseMove"][value="1"]',
                                 'button[name="chooseMoveTarget"][value="2"]',
                                 'button[name="chooseMove"][value="1"]']


def test_listed_struggle_clicks_button_one_as_the_official_client_renders():
    # Live 2026-10-09: the server listed Struggle as the only enabled move; the
    # client renders it as button 1, so clicking value 0 stalled 90 s and crashed.
    page = Page(snapshot())
    page.state['request']['active'][0] = {'moves': [{'id': 'struggle', 'move': 'Struggle', 'target': 'randomNormal',
                                                     'disabled': False}]}
    asyncio.run(BrowserPlayer(page).submit(deepcopy(page.state), 'move 1, move 1'))
    assert page.clicked[0] == 'button[name="chooseMove"][value="1"]'
