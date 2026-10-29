# Answer key: L2_relaxer_template.py — Strategy and template method, level 2 (20 min)

**Pattern focus:** In a template method, the base class owns the invariant steps (state reset, limits, clean-up) and subclasses only fill in hooks. This drill trains you to check where state lives (on the class, the instance, or the run), when each hook runs, what happens on the failure path, and whether an override keeps the base class's contract.

**Good clarifying questions:** Is one relaxer reused for many structures? The docstring says yes. Is the relaxation energy compared across MOFs of different sizes? It's per cell, so bigger cells relax "more"; for a stability descriptor you usually want eV per atom. Who owns the calculator: should `run()` stop a DFT process the worker started? What should happen when a relaxation doesn't converge: an error, or a result flagged as unconverged?

## Blockers (silently wrong results / data loss)
1. **`history` is a class attribute, so every relaxer shares one list, and the relaxation energy is wrong from the second structure on.** `history = []` in the class body is shared by every `BaseRelaxer` instance and subclass, and nothing clears it. `run()` returns `self.history[0] - energy`, so every structure after the first is measured against the *first* structure's starting energy. In the checks, a structure that relaxes by 1.8 eV is reported as 7.5 eV. The list also grows by one float per step for the life of the worker. *Fix:* set `self.history = []` at the start of `run()`. Doing it in `__init__` isn't enough, because one relaxer handles many structures.

2. **`SteepestDescent.step` overwrites the caller's coordinates.** `positions += self.step_size * forces` modifies the array in place, and on the first step that array is the one the caller passed to `run()`. After `relaxer.run(structure.positions)`, the structure's starting geometry is gone. If the pipeline stores (initial, relaxed) pairs, for example as training data for the ML potential, both halves are the relaxed geometry. *Fix:* `return positions + self.step_size * forces`, and have `run()` start from a copy, `np.array(positions, dtype=float)`, so no future hook can do this.

3. **`MLRelaxer` changes what "converged" means.** The base contract, as the PR states it, is "the largest force on any atom is below fmax". `MLRelaxer.converged` uses the *mean* per-atom force instead. In a 400-atom MOF where one atom has 1 eV/Å and the rest are relaxed, the mean is 0.0025 eV/Å, so a structure with a badly placed atom counts as converged. Downstream code (DFT single points, stability filters) relies on the base contract. The noise argument is real, but averaging hides exactly the local defects you care about. *Fix:* keep the max-force criterion and use a looser `fmax` for the ML pre-relaxation (e.g. 0.1 eV/Å), then refine with DFT if needed.

## Major
4. **There's no step limit.** `while not self.converged(forces)` has no `max_steps`. A noisy ML potential hovering just above `fmax`, a step size that oscillates, or a structure that keeps rearranging loops forever, holding a worker and a DFT licence or GPU. *Fix:* a `max_steps` argument. When it's reached, raise a specific `RelaxationNotConverged` (or return a result flagged as unconverged) so the screen can record it and move on.

5. **`teardown()` isn't in a `finally`.** If the calculator raises (SCF failure, lost node) or `step` raises, `run()` exits before `self.teardown()`. The DFT process keeps running and holding cores and a licence, once for every structure that fails. *Fix:* `try: ... finally: self.teardown()` around the loop.

6. **`setup()` is called from `__init__`, not from `run()`.** There are three consequences:
   - It runs once per relaxer, but the worker calls `run()` once per structure. After the first `run()`, `teardown()` has stopped the calculator, so the second `run()` talks to a stopped DFT process (in the checks: `RuntimeError: calculator is not running`).
   - `DampedDynamics` keeps the previous structure's `velocities`. For a structure with a different number of atoms that gives `ValueError: operands could not be broadcast together with shapes (2,3) (5,3)`. With the same number of atoms, the new relaxation silently starts with a kick from the old one.
   - The hook runs before the subclass's `__init__` body. No current `setup` reads a subclass attribute, but the first one that does (say `self.dt`) gets `AttributeError`.

   *Fix:* call `setup()` at the start of `run()`, and `teardown()` in its `finally`.

## Minor
7. **`BaseRelaxer` isn't abstract.** `step()` raises `NotImplementedError`, so `BaseRelaxer(calc)` builds fine (and starts the calculator), and only fails on the first step. *Fix:* make it an `ABC` with `@abstractmethod` on `step`.

## Red herrings (say why they're fine)
- **The sign in `positions + step_size * forces`.** Gradient descent is usually written `x -= lr * grad`, but forces are minus the energy gradient, so stepping *along* the forces goes downhill. The problem on that line is the in-place `+=` (#2), not the sign.
- **`DampedDynamics.step` returns `positions + self.dt * self.velocities`.** That creates a new array, so this subclass doesn't have the aliasing bug; only `SteepestDescent` does.

## Model 1-minute summary
"Three things silently give wrong numbers. `history` is a class attribute, so every relaxation in a worker shares one list, and from the second structure on, the relaxation energy is measured from the first structure's starting energy. `SteepestDescent` updates positions in place, which overwrites the caller's starting geometry. And `MLRelaxer` redefines convergence as the mean force, so one badly placed atom in a big cell passes; I'd use a looser `fmax` instead. Then the template itself. There's no step limit, so a relaxation can spin forever. Teardown isn't in a `finally`, so a crash leaks a DFT process. And `setup` runs in `__init__` rather than per run, so the worker's second structure gets a stopped calculator and stale velocities. The base class should copy the input, reset its state, call setup and teardown around the loop, and enforce `max_steps`."

## Reference fix
`L2_relaxer_template_fixed.py`:
- `run()` owns the invariants. It copies the input, resets `history`, calls `setup()`, enforces `max_steps` (raising `RelaxationNotConverged`) and calls `teardown()` in a `finally`.
- `BaseRelaxer` is an ABC with an abstract `step`, and `history` is per instance and per run.
- `SteepestDescent.step` returns a new array.
- `MLRelaxer` keeps the base max-force criterion, with a default `fmax` of 0.1 eV/Å.

Checks: `uv run --with pytest --with numpy pytest python_patterns/checks/test_06_strategy_template.py -k L2 -v`
