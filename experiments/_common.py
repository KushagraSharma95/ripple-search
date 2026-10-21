import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

try:
    from threadpoolctl import threadpool_limits  # noqa: F401
except ImportError:  # pragma: no cover
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(limits=None):
        yield


def save(name: str, payload: dict) -> None:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {path}")
