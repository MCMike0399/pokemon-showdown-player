# Watching a running campaign

The read-only viewer renders battle animations inside the page and follows each
new game. It uses the existing player's relay, so private battles work without a
second Showdown login or changed room privacy. Sound starts muted.

Run on the machine holding the harness and its data:

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
