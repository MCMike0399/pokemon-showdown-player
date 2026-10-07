import asyncio, os
from pathlib import Path
for line in (Path(__file__).parent/".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k,v=line.split("=",1); os.environ.setdefault(k.strip(), v.strip())
from ps_client import PSClient

async def main():
    c = PSClient()
    await c.connect()
    print("challstr:", bool(c.challstr))
    try:
        r = await c.login()
        print("login:", r)
    except Exception as e:
        print("login ERR:", e)
    await c.join("lobby")
    await asyncio.sleep(3)
    print("user:", c.user, "rooms:", list(c.rooms)[:6])
    print("tail:", [l[:80] for l in (await c.read("lobby", 5))])
    await c.close()

asyncio.run(main())
