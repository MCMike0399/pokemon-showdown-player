import json

from battle_state import legal_choices, observe
from harness import pack_set


def request():
    mons = [{"ident": f"p2: {s}", "details": s + ", L50", "condition": "100/100", "active": i < 2}
            for i, s in enumerate(("Pelipper", "Swampert", "Raichu", "Incineroar", "Rillaboom", "Archaludon"))]
    return {"rqid": 1, "side": {"id": "p2", "pokemon": mons}, "active": [
        {"moves": [{"move": "Hurricane", "target": "normal", "pp": 10}, {"move": "Tailwind", "target": "allySide", "pp": 10}]},
        {"canMegaEvo": True, "moves": [{"move": "Waterfall", "target": "normal", "pp": 10}, {"move": "Helping Hand", "target": "adjacentAlly", "pp": 10}]}]}


def test_doubles_targets_joint_actions_not_truncated():
    choices = legal_choices(request())
    assert "move 1 1, move 1 2 mega" in choices
    assert "move 2, move 2 -1" in choices
    assert "move 1 -2, move 1 -1" in choices
    assert len(choices) > 60
    assert "switch 3, switch 3" not in choices


def test_preview_leads_and_bring_four():
    req = request()
    req.update(teamPreview=True, maxChosenTeamSize=4)
    choices = legal_choices(req)
    assert len(choices) == 360
    assert "team 6,5,4,3" in choices


def test_one_mega_and_one_tera():
    req = request()
    for a in req["active"]:
        a.update(canMegaEvo=True, canTerastallize="Water")
    choices = legal_choices(req)
    assert all(c.count("mega") <= 1 and c.count("terastallize") <= 1 for c in choices)
    assert any("mega" in c for c in choices)


def test_forced_switch_partial_and_insufficient_bench():
    req = request()
    req.pop("active")
    req["forceSwitch"] = [False, True]
    assert legal_choices(req) == [f"pass, switch {n}" for n in range(3, 7)]
    req["forceSwitch"] = [True, True]
    for mon in req["side"]["pokemon"][3:]:
        mon["condition"] = "0 fnt"
    assert set(legal_choices(req)) == {"switch 3, pass", "pass, switch 3"}
    req["side"]["pokemon"][2]["condition"] = "0 fnt"
    assert legal_choices(req) == ["pass, pass"]


def test_wait_trapped_disabled_and_fainted():
    req = request()
    req["wait"] = True
    assert legal_choices(req) == []
    req.pop("wait")
    req["active"][0]["trapped"] = True
    req["active"][0]["moves"][0]["disabled"] = True
    assert all(c.startswith("move 2,") for c in legal_choices(req))
    req["side"]["pokemon"][0]["condition"] = "0 fnt"
    assert all(c.startswith("pass,") for c in legal_choices(req))


def test_locked_two_turn_move_does_not_choose_another_target():
    req = request()
    req["active"][0] = {"trapped": True, "moves": [{"id": "electroshot", "move": "Electro Shot"}]}
    assert all(c.startswith("move 1,") for c in legal_choices(req))


def test_p2_observation_clears_conditions_and_keeps_both_opponents():
    log = ["|switch|p1a: Bird|Pelipper, L50|100/100", "|switch|p1b: Frog|Swampert, L50|100/100",
           "|-weather|RainDance", "|-weather|none", "|-sidestart|p1: Them|move: Tailwind",
           "|-sideend|p1: Them|move: Tailwind", "|-fieldstart|move: Trick Room", "|-fieldend|move: Trick Room",
           "|-damage|p1b: Frog|35/100", "|-boost|p1b: Frog|atk|2", "|turn|2",
           "|move|p1a: Bird|Hurricane|p2a: Bird", "|showteam|p1|Pelipper||focussash|drizzle|hurricane,protect|Modest||||||"]
    state = observe(request(), log)
    assert state["side_id"] == "p2"
    assert [m["species"] for m in state["opp_actives"]] == ["Pelipper", "Swampert"]
    assert state["opp_actives"][1]["condition"] == "35/100"
    assert state["opp_actives"][1]["boosts"]["atk"] == 2
    assert not state["weather"] and not state["field"] and not state["hazards"]["theirs"]
    assert state["opp_team_sheet"][0]["moves"] == ["hurricane", "protect"]
    assert state["history"][0]["side"] == "theirs"
    assert state["my_actives"][0]["species"] == "Pelipper"


def test_private_request_lines_are_not_opponent_observations():
    state = observe(request(), ['|request|' + json.dumps({"side": {"id": "p1", "pokemon": [{"species": "Mew", "item": "secret"}]}})])
    assert state["opp_revealed"] == []
    assert state["opp_team_sheet"] == []


def test_packed_team_preserves_ivs_and_tera():
    parts = pack_set({"species": "Pelipper", "moves": ["Protect"], "ivs": {"atk": 0}, "teraType": "Grass"}).split("|")
    assert parts[8] == ",0,,,,"
    assert parts[11] == ",,,,,Grass"
