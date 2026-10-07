import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import TeamStore, pack_team, pack_set

SETS = [
 {"name":"Pelipper","species":"Pelipper","item":"Focus Sash","ability":"Drizzle","moves":["Weather Ball","Tailwind","Hurricane","Protect"],"nature":"Modest","level":50,"evs":{"hp":31,"def":1,"spa":5,"spd":18,"spe":11}},
 {"name":"Archaludon","species":"Archaludon","item":"Leftovers","ability":"Stamina","moves":["Electro Shot","Dragon Pulse","Flash Cannon","Protect"],"nature":"Modest","level":50,"evs":{"hp":27,"def":5,"spa":5,"spd":10,"spe":19}},
]

tmp = Path(tempfile.mkdtemp())/"teams.json"
s = TeamStore(tmp)
print("empty:", s.list())
s.create("Rain VGC", SETS, "gen9championsvgc2026regmc")
print("after create:", s.list())
print("get.format:", s.get("Rain VGC")["format"])
packed = s.packed("Rain VGC")
print("packed len:", len(packed), "| members:", packed.count("]")+1)
assert pack_set(SETS[0]).startswith("Pelipper|")
s.update("Rain VGC", fmt="gen9vgc2025regi", new_name="Rain VGC v2")
print("after update:", s.list(), s.get("Rain VGC v2")["format"])
print("delete:", s.delete("Rain VGC v2"), "->", s.list())
print("CRUD OK")
