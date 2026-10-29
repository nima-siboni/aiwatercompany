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

# on a general note, what is suppoised to be called from outside? screen? if yes I would pass all the paramters to it for reproducibility.

# in general this code has a strong reproducibility challenges.

# I am not 100% sure about the concurrent.futures and sub processes syntax.

def load_cache():
    global _cache # in general I dont like this global variables. a constant is still ok but a global variable is not. Better use singelton
    if os.path.exists(CACHE_FILE):
        _cache = json.load(open(CACHE_FILE))


def save_cache(): # why the hell this function does not get its argumetns? Make it a standalone function and put it in a file_utils.py
    with open(CACHE_FILE, "w") as f: 
        json.dump(_cache, f)



# is it possible to have more than one simulation for a unique key, if yes what should be done with the duplicates?
def cache_key(mof_id, params):
    # a one line function?
    return mof_id + str(params)

def run_simulation(mof_id, params={"temperature": 298, "pressure": 1.0, "cycles": 5000}):
    # I dont like this implicit defaults. It is not a good practice, specially that we are not writing the parameters anywhere.
    # Also most probably there will be other ways of stopping than the cycles count. This way we are locked in.

    key = cache_key(mof_id, params)
    if key in _cache:
        return _cache[key] # do we need to return the 

    cmd = (
        f"gcmc_sim --structure {mof_id}.cif "
        f"--T {params['temperature']} --P {params['pressure']} --cycles {params['cycles']}"
    )
    for attempt in range(3): # why 3? what happens if there is no success? shal
        try:
            out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600) # what the fuck!?
            uptake = float(out.stdout.strip().split("\n")[-1]) # this is dangerous. If it is possible I will make it more robust: make the  the gcmc_sim write the results in a formatted file.
            # what will happen if there is no result, what would be saved in the ky
            # can we distini=guish between different reasons why there is no results: this helps us understand whether we should retry or not, e.g. out of memory is different from large dt.
            break
        except:
            time.sleep(1)

    _cache[key] = uptake
    save_cache()
    return uptake


def screen(mof_ids, params=None, n_workers=32): # I dont like that different mofs are simulated with the same params. Also in general it is not a good idea to just run the sim with one set of paramters.
    load_cache()
    with ThreadPoolExecutor(n_workers) as ex:
        if params:
            results = list(ex.map(lambda m: run_simulation(m, params), mof_ids))
        else:
            results = list(ex.map(run_simulation, mof_ids)) 
    return dict(zip(mof_ids, results))
