# Players: one contract, two transports

`ml/players.py` defines `IPlayer`, the contract for anything that plays Showdown
for our account. It's a transport: log in, build or select the team, find a ladder
battle, list active battles, enter a room, read a battle's state, submit a
choice, timer, forfeit, leave, close. It never decides what to play. Turns are
always decided by the local model (search agent for moves, Brain for team
preview), whichever transport carries them.

| Implementation | How | Role |
|---|---|---|
| `PlaywrightPlayer` (`ml/browser.py`, alias `BrowserPlayer`) | the official classic client in Chrome, programmatic clicks | default for everything |
| `McpPlayer` (`ml/players.py`) | OpenClaw's Pokemon Showdown MCP harness (`ps_mcp_server.py`, the tools OpenClaw loads), driven as an MCP client over stdio, with its own websocket connection for the same account | fallback for edge cases |

The live runner (`scripts/browser_play.py`) goes through the interface:
`active_battles`, `enter_room` and `find_battle` (which uses `select_team`). The
game loop (`PlaywrightPlayer.play`) uses `snapshot`, `submit` and `set_timer`.

## When the MCP harness is used

`--fallback mcp` (the default; `--fallback none` disables it). The harness
process starts and logs in only when it's first needed. It refuses to act if it
isn't logged in as the configured account.

1. **Lost choice.** Chrome marked our choice sent, but the server's timer
   warnings say it is still waiting for us (`_recover_lost_choice`).
   - Attempt 1: Chrome re-sends `/choose <choice>|<rqid>`.
   - Attempts 2 and 3: the MCP harness logs in (checked every time), joins
     the room, checks that the server's request is still that `rqid`, and sends
     the same choice with `ps_choose`. If the harness fails, Chrome re-sends
     instead. After three attempts the runner stops with an error and launchd
     restarts it.
   - One delivery through the harness is bounded to 12 s (a cold start plus
     login takes about 3-4 s), so it can't eat the turn clock. A dead or hung
     harness process is restarted on its next use.
2. **Chrome can't submit.** If clicking fails three times for one request (a
   control that never becomes clickable, a request changing mid-click), the
   third time the MCP harness sends the choice instead of the runner crashing.
   If the server has already moved on, the runner decides the new request.

The harness never sends for a request the server has moved past. A fallback
failure is logged (`fallback_failed`) and is never worse than having no
fallback. Events: `fallback_submit`, `fallback_skipped`, `fallback_failed`.

## Drills

```sh
touch data/ml/drill-lost-choice                  # next preview choice is lost; Chrome's re-send recovers it
echo fallback > data/ml/drill-lost-choice        # Chrome's re-send is also suppressed: only MCP can recover it
```

Each drill fires once, at the next team preview, and deletes its file.
