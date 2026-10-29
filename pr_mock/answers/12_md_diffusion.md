# Answer key: 12_md_diffusion.py

**Good clarifying questions:** How long do 2 million steps take with the MLIP on these cells,
compared with the 48-hour limit? How many CO2 molecules are in a cell (that sets the
statistics)? Do we want D at one loading or as a function of loading? Has the potential been
checked for energy drift in NVE? Are the guests reliably tagged 1, and nothing in the framework?

## Blockers

1. **`TIMESTEP = 1.0` is in ASE time units, which are 10.18 fs.** ASE measures time in
   Å·√(amu/eV), so `timestep=1.0` integrates with 10.18 fs steps. That is unstable for
   molecules with C–H or C=O vibrations: a harmonic C–H stretch (period 11 fs) blew up within
   200 steps at 10.18 fs and stayed stable at 1 fs. Even a stable run would be wrong three
   ways:
   - 2 million steps would be 20 ns, not 2 ns;
   - the analysis still assumes 1 fs per step, so D would come out 10.18× too large;
   - a 10 fs step is far too long for these vibrations, so the dynamics themselves are wrong.
   *Fix:* `timestep=1.0 * units.fs`, and take the analysis time step from
   `dyn.dt / units.fs` rather than from a second constant.

2. **The stored trajectory is wrapped** (`dyn.attach(atoms.wrap, ...)`). The MSD needs
   unwrapped coordinates. Wrapping every 100 steps moves atoms by a whole cell length between
   frames: a 7.8 Å jump in a 7.2 Å test cell. In a Brownian test (D = 1e-5 cm²/s, a 25 Å box,
   2 ns), the wrapped MSD flattened out (382 Å² at 2 ns against 1,180 unwrapped), and the PR's
   D came out 14× too small. *Fix:* never wrap the positions you analyse. Write a wrapped copy
   for visualisation if you need one, or unwrap with cumulative minimum-image displacements.

3. **The MSD is averaged over x, y and z, then divided by 6** (`diffusion_coefficient`).
   `.mean(axis=(1, 2))` averages the three components, so `slope / 6` is D/3. On the same
   trajectory, the PR's D was exactly 0.333 of the correct value. *Fix:* sum the components
   (`((pos - pos[0]) ** 2).sum(-1).mean(-1)`), then divide the slope by 6 (by 2d in general:
   2 for diffusion along 1-D channels).

4. **Resuming destroys the run** (`run_md`). The PR promises that resubmitted tasks continue
   from their trajectory, but four things go wrong:
   - **The trajectory is overwritten.** ASE's `append_trajectory` defaults to False, so
     constructing `Langevin` with the same path truncates the file. In the check, a 21-frame
     trajectory had 6 frames after a "resume" that ran 100 steps.
   - **The full count runs again.** `dyn.run(N_STEPS)` starts the whole 2 million steps from
     scratch.
   - **The velocities are redrawn.** `MaxwellBoltzmannDistribution` throws away the
     equilibrated velocities read from the trajectory.
   - **A fragment is reported as the result.** The `finally` then analyses whatever fragment
     exists and overwrites `_D.txt` with it, as if the run had finished.
   *Fix:* `append_trajectory=True`, run only the remaining steps, skip the velocity draw on
   resume, and only write D for complete runs.

## Major

5. **The `finally` doesn't run when Slurm hits the time limit.** Slurm sends SIGTERM, which
   ends Python without running `finally` blocks. In the check: exit code −15, and the
   `finally` never ran. Runs that hit 48 hours therefore never write D. *Fix:* install a
   SIGTERM handler that raises `SystemExit`, ask Slurm for an early signal
   (`--signal=B:USR1@300`), and run the analysis as a separate step once a run is complete.
6. **A strong Langevin thermostat for a transport property.** `0.01 / units.fs` is 10 ps⁻¹,
   applied to every atom, guests included, and that friction slows diffusion. A back-of-the-
   envelope check: the thermostat alone limits a free CO2 molecule to D = kT/(mγ) ≈ 5.7e-5
   cm²/s, within a factor of about 5 of the D we're measuring. *Fix:* equilibrate with
   Langevin, then measure in NVE, or with a weak global thermostat (CSVR, Nosé–Hoover).
7. **The error bar comes from the polyfit covariance.** That assumes the MSD points are
   independent, but they come from one trajectory and one time origin, so neighbouring points
   are almost identical. Over 40 independent synthetic runs, the reported error was 55× smaller
   than the real spread of D. *Fix:* average over many time origins, and take the error from
   independent blocks or replicas.
8. **The array index is off by one.** `--array=1-50` gives task IDs 1 to 50, so `mofs[task]`
   skips the first MOF, and task 50 raises IndexError. *Fix:* `--array=0-49` (or
   `mofs[task - 1]`), and a manifest written at submission time, so a file added later doesn't
   shift the list.
9. **The whole trajectory is loaded into memory.** `read(traj, index=":")` builds 20,000 Atoms
   objects (2 million steps / 100) of a few thousand atoms each: gigabytes of Python objects to
   keep a few hundred carbon positions. *Fix:* iterate over the `Trajectory`, and keep only the
   guest carbons.

## Minor

10. **One time origin, fitted from t = 0.** The fit includes the ballistic regime at short
    times and the noisy tail at long lags, and frame 0 comes straight after the velocity draw,
    with no equilibration discarded. Use many time origins, fit only the linear regime (lags up
    to about a tenth of the run), and discard the first 100 ps or so. In the check, multiple
    origins recovered 9.6e-6 against a true 1e-5.
11. **The result file is bare.** It records neither the run length, the number of molecules nor
    the temperature, so a fragment can't be told apart from a full run.

## Red herrings (say why they're fine)

- **`friction=0.01 / units.fs`** is the documented way to write the friction. The unit is
  right; the issue for D is the value (6).
- **`temperature_K=300`** is the right keyword. The old `temperature=` was in eV.
- **No `Stationary()` call:** ASE's Langevin keeps the centre of mass fixed by default
  (`fixcm=True`).
- **Carbon positions as a proxy for each CO2** are fine: the carbon sits at the centre of
  O=C=O.
- **`to_cm2_s = 1e-16 / 1e-15`** is correct: 1 Å²/fs = 0.1 cm²/s.
- **A float32 MACE calculator for MD** is fine; MACE recommends float32 for MD.
- **`sorted(glob(...))`** gives a deterministic order. The problems are the indexing (8) and
  listing the files at run time.

## Model 1-minute summary
"The D values will be wrong by large factors, in both directions. The time step of 1.0 is in
ASE units, 10.18 fs, so the dynamics are unstable for these molecules, and the analysis then
assumes 1 fs per step. The trajectory is wrapped by an observer, so the MSD flattens; in a
synthetic test that made D 14 times too small. The MSD averages over x, y and z before dividing
by 6, which is another factor of 3. The resume logic overwrites the trajectory and reruns the
full count. And the `finally` meant to save results doesn't run on Slurm's SIGTERM. After those:
the Langevin friction slows the diffusion we're trying to measure, the polyfit error bar is 55
times too small, and the array index is off by one. I'd ask for a test on Brownian motion with a
known D, and an NVE drift check of the potential, before running 50 MOFs."
