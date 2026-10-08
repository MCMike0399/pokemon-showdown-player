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
entry or reconnection, then pushes changes over server-sent events (SSE). This is
a one-way persistent HTTP stream. Each browser has its own renderer and replay
selection; slow/disconnected viewers receive the latest state without an event
backlog. A viewer restart recovers from the saved relay and read-only SQLite data.
Campaign scores are checked every two seconds, relay frames every 250 ms. These
intervals do not change the player's existing publication rate.

Live entry, reconnection and returning to a background tab seek to the current
match position. Normal new events animate, with automatic catch-up if the renderer
falls more than a turn behind or accumulates over 80 events. Sound stays muted by
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
Terminal scores come from SQLite rather than a runner's counter or a hardcoded
account name. When no live frame is available, the viewer uses completed recorded
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
