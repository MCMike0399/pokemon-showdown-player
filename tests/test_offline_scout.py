"""Offline checks for scouting an opponent's earlier public games (search/scout.py)."""
import asyncio
import random
from pathlib import Path

import pytest

from search.scout import SimScout, StaticScout, apply_facts, extract_facts, merge_facts, sheet_fits

ROOT = Path(__file__).resolve().parents[1]

OLD = ['|player|p1|Someone|1', '|player|p2|Rival Player|2', '|gametype|doubles',
       '|poke|p1|Politoed, L50, M|', '|poke|p1|Archaludon, L50, M|',
       '|poke|p2|Charizard, L50, M|', '|poke|p2|Incineroar, L50, M|', '|poke|p2|Rillaboom, L50, M|',
       '|poke|p2|Garchomp, L50, M|', '|poke|p2|Sneasler, L50, F|', '|poke|p2|Kingambit, L50, M|',
       '|start',
       '|switch|p1a: Politoed|Politoed, L50, M|167/167', '|switch|p1b: Archaludon|Archaludon, L50, M|167/167',
       '|switch|p2a: Charizard|Charizard, L50, M|100/100', '|switch|p2b: Incineroar|Incineroar, L50, M|100/100',
       '|turn|1',
       '|detailschange|p2a: Charizard|Charizard-Mega-Y, L50, M',
       '|-mega|p2a: Charizard|Charizard|Charizardite Y',
       '|move|p2b: Incineroar|Fake Out|p1a: Politoed', '|move|p2a: Charizard|Heat Wave|p1a: Politoed|[spread] p1a,p1b',
       '|-enditem|p2b: Incineroar|Sitrus Berry|[eat]',
       '|turn|2', '|move|p2b: Incineroar|Parting Shot|p1a: Politoed', '|win|Someone']


def test_extract_and_merge_earlier_games():
    facts = extract_facts(OLD, 'rivalplayer')
    assert facts['six'][:2] == ['charizard', 'incineroar']
    assert facts['mons']['charizard']['item'] == 'charizarditey' and 'heatwave' in facts['mons']['charizard']['moves']
    assert facts['mons']['incineroar']['item'] == 'sitrusberry'
    assert set(facts['mons']['incineroar']['moves']) == {'fakeout', 'partingshot'}
    assert extract_facts(OLD, 'nobody') is None
    six = ['Charizard', 'Incineroar', 'Rillaboom', 'Garchomp', 'Pelipper', 'Kingambit']
    merged = merge_facts([facts], six, min_overlap=4)
    assert merged['incineroar']['item'] == 'sitrusberry' and merged['incineroar']['games'] == 1
    # A different team core gives nothing.
    assert merge_facts([facts], ['Charizard', 'Venusaur', 'Pelipper', 'Tyranitar', 'Gholdengo', 'Dragonite'], 4) == {}


def test_current_game_reveals_win_over_scouted_facts():
    scouted = {'moves': ['fakeout', 'knockoff', 'partingshot', 'flareblitz'], 'item': 'sitrusberry', 'ability': 'intimidate'}
    obs = apply_facts({'moves': ['uturn'], 'item': None, 'ability': None}, scouted, item_known=False, item_gone=False)
    assert obs['moves'][0] == 'uturn' and len(obs['moves']) == 4 and obs['item'] == 'sitrusberry'
    known = apply_facts({'moves': [], 'item': 'safetygoggles', 'ability': None}, scouted, item_known=True, item_gone=False)
    assert known['item'] == 'safetygoggles'
    assert sheet_fits({'moves': ['fakeout', 'uturn']}, ['uturn']) and not sheet_fits({'moves': ['fakeout']}, ['uturn'])


def test_sim_scout_is_seeded_and_bounded():
    sets = [{'species': 'Incineroar', 'item': 'Sitrus Berry', 'ability': 'Intimidate',
             'moves': ['Fake Out', 'Knock Off', 'Parting Shot', 'Flare Blitz']}]
    a = asyncio.run(SimScout(sets, 7, coverage=1.0, p_item=1.0, p_wrong_item=0.0, p_move=1.0).facts('x', []))
    assert a['incineroar']['item'] == 'sitrusberry' and len(a['incineroar']['moves']) == 4
    assert asyncio.run(SimScout(sets, 7, coverage=0.0).facts('x', [])) == {}


@pytest.mark.skipif(not (ROOT / 'cache' / 'usage').exists(), reason='usage statistics not downloaded')
def test_agent_worlds_use_scouted_items():
    from search.agent import SearchAgent
    from search.tracker import Tracker

    async def run():
        agent = SearchAgent(worlds=1, scout={'trust': 1.0})
        agent.scout_source = StaticScout({'Rival Player': {'rillaboom': {'moves': ['woodhammer'], 'item': 'assaultvest',
                                                                        'ability': None, 'sheet': None, 'games': 2}}})
        ctx = {'request': {'teamPreview': True, 'side': {'id': 'p1', 'pokemon': []}}, 'choices': ['team 1234'],
               'public_log': OLD[:12]}
        await agent.decide(ctx)
        await asyncio.sleep(0)
        agent._collect_scout()
        tracker = Tracker('p1').feed(OLD[:17])
        sides = [agent._their_side(tracker, random.Random(k)) for k in range(30)]
        await agent.close()
        return agent.scout_facts, sides

    facts, sides = asyncio.run(run())
    assert 'rillaboom' in facts
    rilla = [m['set'] for side in sides for m in side if m['set']['species'] == 'Rillaboom']
    assert rilla and all(s['item'] == 'assaultvest' and 'woodhammer' in s['moves'] for s in rilla)
