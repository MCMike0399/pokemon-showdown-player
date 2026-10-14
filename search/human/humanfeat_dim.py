"""The action feature width of search/human/humanfeat.cjs, from its vocabulary (no node needed)."""
import json
from pathlib import Path

_V = json.loads((Path(__file__).with_name('vocab.json')).read_text())
ACTION_DIM = 2 * 31 + 8
DIM = ACTION_DIM + 2 * (len(_V['moves']) + 1 + len(_V['species']) + 1)
NM = len(_V['moves']) + 1
NS = len(_V['species']) + 1
