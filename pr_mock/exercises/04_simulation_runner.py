"""
PR #171: Run GCMC adsorption simulations in parallel and cache results

We screen thousands of candidate MOFs with GCMC to label data for the
surrogate model. This adds a parallel runner and an on-disk cache so we
don't re-run simulations we've already done.
"""
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

CACHE_FILE = "results_cache.json"
_cache = {}

def load_cache():
    global _cache
    if os.path.exists(CACHE_FILE):
        _cache = json.load(open(CACHE_FILE))


def save_cache():
    with open(CACHE_FILE, "w") as f: 
        json.dump(_cache, f)


def cache_key(mof_id, params):
    # a one line function?
    return mof_id + str(params)

def run_simulation(mof_id, params={"temperature": 298, "pressure": 1.0, "cycles": 5000}):

    key = cache_key(mof_id, params)
    if key in _cache:
        return _cache[key]

    cmd = (
        f"gcmc_sim --structure {mof_id}.cif "
        f"--T {params['temperature']} --P {params['pressure']} --cycles {params['cycles']}"
    )
    for attempt in range(3):
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
            uptake = float(out.stdout.strip().split("\n")[-1])
            break
        except:
            time.sleep(1)

    _cache[key] = uptake
    save_cache()
    return uptake


def screen(mof_ids, params=None, n_workers=32):
    load_cache()
    with ThreadPoolExecutor(n_workers) as ex:
        if params:
            results = list(ex.map(lambda m: run_simulation(m, params), mof_ids))
        else:
            results = list(ex.map(run_simulation, mof_ids)) 
    return dict(zip(mof_ids, results))
