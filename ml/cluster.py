"""Optional host-wide admission budget; independent of model and experiment policy."""
import json
import math
import os
from pathlib import Path
import time


def background_budget(path=None, now=None):
    configured = path or os.environ.get('BURBUJA_BACKGROUND_BUDGET')
    if not configured:
        return None
    try:
        value = json.loads(Path(configured).read_text())
        at, workers = value['at'], value['maxWorkers']
        if type(at) not in (int, float) or not math.isfinite(at) or type(workers) is not int or not 0 <= workers <= 64:
            raise ValueError('invalid budget')
        age = (time.time() if now is None else now)-at
        if not -5 <= age <= 90:
            raise ValueError('stale budget')
        return workers
    except (OSError, ValueError, KeyError, TypeError):
        # Opt-in consumers defer new background work if the coordinator disappears.
        return 0
