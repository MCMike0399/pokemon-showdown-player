# Watching a running campaign

The read-only viewer renders battle animations inside the page and follows each
new game. It uses the existing player's relay, so private battles work without a
second Showdown login or changed room privacy. Sound starts muted.

Run the retained viewer on an always-on machine. Devices connected to the same
Tailscale network need only a browser; they do not run a player, copy campaign
files, or keep an SSH tunnel open. For HTTPS access, point a hostname such as
`pokemon.example.com` at the server's current Tailscale address. Configure Caddy
to listen only on that address and proxy to IPv4 loopback port 18489. Use DNS-01
certificate validation and include the hostname in your certificate/DNS renewal
setup. Keep this route private to the tailnet.

The server maintains one current campaign/battle snapshot from the existing
player's atomic relay files. `/api/events` sends a full snapshot immediately on
entry or reconnection, then pushes changes over server-sent events (SSE). The
viewer opts into `/api/events?deltas=1`: subsequent messages carry only newly
appended protocol lines and their `log_start` offset. A room change, rewritten
log, or reconnect sends a complete snapshot. Older clients still receive full
snapshots without this parameter. This is
a one-way persistent HTTP stream. Each browser has its own renderer and replay
selection; slow/disconnected viewers receive the latest state without an event
backlog. A viewer restart recovers from the saved relay and read-only SQLite data.
Scores are checked every 500 ms, relay frames every 100 ms. The browser player
observes its existing page every 200 ms during model inference and control waits;
it publishes changed frames atomically without a second Showdown connection.

Opening a game shows its current position instantly; a reconnect to the game
already on screen resumes in place without skipping. After that every turn,
including the final one, plays out in full in Showdown's fast animation mode; animations are never cut on a timer. A frame normally carries one
turn, so the renderer is one turn behind while it animates. With a second turn
queued it switches to hyperfast mode until caught up; only with four or more
queued (or after returning to a background tab that far behind) does it jump to
the start of the newest turn, which still animates. A renderer that stops
advancing for 15 seconds is skipped forward. **Jump to live** skips on demand.
Selected replays animate from the start.

Playback triage. Add `?debug=1` to show live counters over the arena (displayed
vs. live turn, pending steps, speed, skips, resets, main-thread stalls,
reconnects, last turn's play time). Every page also reports these events in
batches to `POST /api/diag`, appended to `data/ml/logs/live-watch-client.jsonl`
(`seek` = a skip during playback with reason and skipped steps, `enter` = the
initial jump into a game, `reset` with reason `room|shrink|rewrite`, `speed`, `jank`, `drained`, `stream-error`).
The server logs each published battle-frame change to
`data/ml/logs/live-watch-frames.jsonl` (`append`, `room`, or `rewrite` with the
first differing line). Both logs keep 5 MB plus one rotated generation. To
reproduce playback without a player, replay a recorded game at its real turn
pace on another port:

    .venv/bin/python scripts/live_watch_sim.py --events data/ml/browser-runs/<run>/events.jsonl [--speed 3] [--loop]
    open 'http://127.0.0.1:18490/?debug=1'

An observation heartbeat marks a stale player feed after ten seconds. Sound stays muted by
default. Select a recent game for playback; **Jump to live** returns to auto-follow.
**Reconnect** replaces the viewer's stream, without touching the player connection.
The sidebar shows an open-ended ladder record without a game cap or target bar.
Overall and recent win rates count wins divided by all verified games in the
corresponding window, including ties. Recent form reads oldest to newest; the
current streak is a lower bound when it fills the available recent window.
Policy updates and opponent samples come from the existing monitoring snapshot.
The performance readout compares the recent window with the full-run average;
it is descriptive and does not claim a model passed an evaluation or improved.

On macOS, install the independent supervised viewer (after freeing its port):

```bash
.venv/bin/python scripts/live_watch_service.py install
.venv/bin/python scripts/live_watch_service.py status
```

The LaunchAgent `dev.pokemon-showdown.live-watch` starts at login and restarts the
viewer if it exits. Logs are in `data/ml/logs/live-watch*.log`. Reinstalling or
uninstalling this agent affects only the viewer, never the campaign/learner:

```bash
.venv/bin/python scripts/live_watch_service.py uninstall
```

The Mac must remain awake. After a reboot, FileVault must be unlocked and the user
logged in before Tailscale and the viewer start. Caddy/certificates/DNS are managed
by the machine's existing ingress configuration, not by this portable repository.
For a new hostname, resolve the server's current address through Tailscale; never
reuse a remembered IP. Add it to the existing certificate/DNS renewal job, and
preserve all sibling Caddy routes. SSE's `text/event-stream` content type is flushed
immediately by Caddy's reverse proxy.

For a manual server on the machine holding the harness and its data:

```bash
.venv/bin/python scripts/live_watch.py
```

On the viewing computer, forward the loopback port through your existing SSH
server alias:

```bash
ssh -N -L 127.0.0.1:18489:127.0.0.1:18489 your-server
```

Then open `http://localhost:18489/`. Keep auto-follow enabled for new matches;
recent games can be replayed in the same panel. `--port` changes the server port.
The server binds IPv4 loopback and serves only its page, scripts and campaign
data endpoints.

Live frames are written atomically under `data/ml/live-watch/` by the MCP battle
observer. The relay excludes requests, chat and authentication messages.
Terminal scores come from SQLite recordings or verified terminal protocol in the
browser ledger, rather than a runner's counter or a hardcoded account name.
Search-agent games intentionally have no PPO episodes; their verified outcomes,
decision counts, perspective and replays come from the ledger. Consecutive
search-agent runs share the record across transport restarts. Rooms are counted
once, and an interrupted game without terminal evidence gets no assigned result.
Reduced ledger summaries are cached until the source file changes. When no live
frame is available, the viewer uses completed recorded
battle logs and labels the view as a replay.

The browser loads Showdown's MIT battle animation engine and public sprite/data
assets. This viewer does not embed the full Showdown application, which refuses
iframe execution. Campaign JSON, game logs, models and relay files stay local
and gitignored.

The `/api/status` and `/api/battle?room=...` JSON endpoints remain available for
read-only diagnostics and older clients. `/api/events?room=...` pins a viewer to a
selected room while continuing to stream campaign status. Relay data excludes
requests, chat and authentication, and no endpoint exposes the raw training database.

Some newer Mega forms are listed in Showdown's battle data before their sprite
files exist. If a Mega sprite fails, the viewer tries available related artwork
and labels the form whose artwork was substituted; battle names, HP and mechanics
remain those of the actual form.

Protocol references: [EventSource and reconnection](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events),
[Caddy streaming responses](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy#streaming).
The client explicitly retries a closed stream after a temporary HTTP error as
well as relying on EventSource's normal dropped-connection retries.

The HTML bootstrap must load Showdown's `data/pokedex-mini.js` before starting
the viewer. It supplies `BattlePokemonSprites`; `data/graphics.js` alone does not
supply animated sprite metadata, so Dex silently selects still PNGs without it.
The known missing-form fallback uses Dex's sprite resolver to preserve GIFs and
image dimensions, while keeping the actual form's cry and battle identity.

## Ladder performance (Elo performance rating)

Live progress is judged by the **performance rating**, not by win rate.
Ladder matchmaking pairs the bot with opponents near its own rating, so the win
rate is pulled toward 50% whether the bot is strong or weak, and a better
policy mostly shows up as tougher opponents. The performance rating is the
rating R at which the sum of Elo expected scores against the opponents actually
faced equals the score actually earned (win = 1, tie = 0.5). It rises when the
bot beats stronger opposition and does not saturate at 50%. The opponent rating
is the one printed on the `|player|` line at game start; our name comes from
`.env` `PS_USERNAME`.

The sidebar panel **Ladder performance** shows:

- the current ladder rating (after the latest game) and its change over the last 50 games;
- **Current**: the performance rating (plus or minus the 95% interval) of the
  current deploy segment, and **Previous** for comparison. A segment is a run of
  consecutive browser runs with the same policy key (a hash of the `search_agent`
  config without volatile fields, plus the preview spec and the team name), so a
  new preview model, team or search setting starts a new segment. Hover a label
  for the changed fields and times;
- the performance (and win rate) over the last 50, 100 and 200 games;
- a sparkline of the rolling-50 performance over all games (at most 200 points),
  with a dashed line where the current segment starts and a dotted reference at its rating.

Read it like this: the interval is about plus or minus 70 at 100 games and 25 at
1,000, so compare the current segment with the previous one only when their
intervals clearly separate; the last 50 is noisy (plus or minus 100). A `*` after
a rating means all wins or all losses, so the value is clamped (100..3000) and has no interval.
The interval ignores the uncertainty in the opponents' own ratings.

`/api/elo` serves the data. It is computed in the server from the browser-run
ledgers (`data/ml/browser-runs/*/events.jsonl`, read incrementally and cached on
file size and mtime), polled by the page every 30 s and after each finished game,
and is not part of the SSE stream. The same numbers are available offline:

    .venv/bin/python search/elo_monitor.py [--runs DIR] [--json]

It prints the segments table (performance, interval, win rate, mean opponent
rating, games, first and last time), the rolling windows and the current rating.
Games are deduped by room (a resumed room keeps its terminal with a log) and
games without an opponent rating are left out of the rating.
