# pokemon-showdown-player

Agent-friendly **live** connection to Pokemon Showdown (not just data lookup).
A small raw-protocol client + an MCP server, so any OpenClaw/Claude agent can
**see** rooms and battles and **interact** (chat, commands, make plays).

Optional **learning brain**: a local PyTorch actor-critic chooses joint VGC
decisions, records experience and learns with PPO or imitation. Includes real
offline Showdown training, current web/team research and outcome-based team
recommendations. [Setup and OpenClaw workflow](docs/ml.md) ·
[Model research and primary sources](docs/ml-research.md).
See [validation and the initial learning experiment](docs/ml-validation.md).

[Technical paper (PDF)](docs/paper/main.pdf) · [LaTeX source and build instructions](docs/paper/README.md):
the system, the data recorded per match, CPU/MPS training, the paired promotion
test, decision-time search and all recorded evidence, with proofs that state
their assumptions.

Always-on use: [OpenClaw decision/team workflow](docs/agent-workflow.md) ·
[Resource-aware CPU/MPS training and daily feeds](docs/continuous-learning.md) ·
[Data sources and licenses](docs/continuous-learning-research.md).

[Programmatic browser play](docs/browser-play.md) uses Playwright clicks and the
same local model for decisions, without OpenClaw or per-turn LLM calls.

[Simulator search agent](docs/search-agent.md) (`--search`) decides each turn by
simulating every action pair in the official engine over opponent sets sampled from
ladder usage statistics: 30 wins vs the PPO policy's 19 on the same 45 real-team games.

## What's here
- `ps_client.py` - async client over wss://sim3.psim.us/showdown/websocket (SockJS).
  Handles challstr, login (assertion), rooms, message buffering and battle state.
- `ps_mcp_server.py` - MCP (stdio) server exposing the tools below.
- `.env` - your account credentials (gitignored, chmod 600). See `.env.example`.
- `selftest.py` - 20-line smoke test (connect -> login -> join lobby -> read).

## MCP tools
| tool | what it does |
|---|---|
| `ps_status` | connection, logged-in user, rooms, battles |
| `ps_login` | connect + log in with the configured account |
| `ps_join` | join a room (lobby, vgc, a battle id, ...) |
| `ps_read` | read recent lines of a room (global = global log) |
| `ps_send` | send chat text or a slash command (/join, /msg, /challenge) |
| `ps_battles` | list battle rooms |
| `ps_battle` | detailed battle state (players, turn, weather, request) |
| `ps_waiting` | battles waiting on a choice |
| `ps_choose` | submit: move N / switch N / team ... |
| `ps_dex` | pokemon / move / item / ability / typechart lookup |

## Quickstart
    cd pokemon-showdown-player
    uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
    cp .env.example .env          # fill your PS_USERNAME / PS_PASSWORD
    .venv/bin/python selftest.py            # smoke test
    .venv/bin/python ps_mcp_server.py       # MCP over stdio

## Account
Supply your own registered Showdown account in `.env`. The repository contains
no account or password. Keep one connection per player and reuse it during a game.

## Notes / gotchas
- The login endpoint wants **pass**, not passwd (https://play.pokemonshowdown.com/api/login).
- Client->server format is ROOMID|TEXT; global commands are |/cmd. See upstream PROTOCOL.md.
- |updateuser|Guest ... means not logged in; |updateuser|<name>|1|... means logged in.
- Keep one long-lived process and reuse the connection (the MCP server does).

## Extending
Add a tool: decorate an async function in ps_mcp_server.py with @mcp.tool().
For battles, PSClient.battle_summary() and current_requests() give the state you need.

## Harness (play like a regular player)

`harness.py` gives a small agent-friendly API:

- **TeamStore** - CRUD teams in `teams.json` (+ PS packed conversion):
      from harness import TeamStore
      ts = TeamStore(); ts.list(); ts.create('Rain VGC', sets, 'gen9championsvgc2026regmc')
      ts.get('Rain VGC'); ts.update('Rain VGC', fmt='...'); ts.delete('Rain VGC'); ts.packed('Rain VGC')
- **Player** - matchmaking + decisions:
      p = Player(); await p.login()
      await p.ladder('gen9randombattle', ts.packed('Rain VGC'))   # or .challenge(user, fmt)
      room = await p.wait_for_request(timeout=90)
      p.legal_choices(room)     # ['move 1','move 2','switch 2']
      await p.choose(room, 'move 1')
      await p.play(room, decide_fn)   # decide_fn(ctx) -> choice ; ctx has turn/choices/request/battle

- **CLI**:
      .venv/bin/python harness.py teams list
      .venv/bin/python harness.py teams packed 'Rain VGC'
      .venv/bin/python harness.py play gen9randombattle

### Tests
      .venv/bin/python -m pytest -q                 # offline rules, learning, research + simulator
      .venv/bin/python tests/test_teams.py            # team CRUD (offline)
      .venv/bin/python tests/test_play.py             # live random battle
      .venv/bin/python mcp_selftest.py                # MCP tools + live login

### MCP tools for the harness
`ps_team_list/get/create/update/delete/packed`, `ps_ladder`, `ps_choices`, `ps_result`
(plus the connection tools above). So an agent can: create a team -> ladder -> read `ps_waiting`,
-> `ps_choices` -> `ps_choose` -> `ps_result`, entirely through MCP.

### Offline PPO notebook

Run `./scripts/run_notebook.sh` and open `notebooks/01_offline_ppo.ipynb` with the
**Pokemon Offline (M5 Pro)** kernel. It compares learning-rate and entropy
settings, completed-episode rewards, CPU/Metal throughput, and replay-trained
scouting in isolated offline runs. See [setup and operation](docs/offline-ppo-notebook.md)
and [experiment sources](docs/offline-ppo-experiments-research.md).
