"""Run every experiment in sequence (each in its own process for isolation)."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    "exp1_scaling_n.py",
    "exp2_dimensionality.py",
    "exp3_hebbian_adaptation.py",
    "exp4_thermo_cache.py",
    "exp5_real_text.py",
    "exp6_temporal_consolidation.py",
]

for exp in EXPERIMENTS:
    print(f"\n=== {exp} ===", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / exp)], cwd=HERE)
    print(f"=== {exp} done in {time.time()-t0:.0f}s (rc={r.returncode}) ===", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
