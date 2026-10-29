# 8. Simulation workflows: DFT outputs, MD and the cluster

Simulation code fails quietly. A parser reads a run that was killed halfway, a label comes from
the wrong energy, or an MD driver passes 300 where the library expects eV, and every number
downstream is still a float. Most of these bugs sit in library defaults; the cluster adds
off-by-one array indices, jobs that leave the queue unfinished, and restarts that lose output.
VASP files below are pymatgen's test fixtures; Slurm and VASP were not run (claims cite docs).

Outputs come from Python 3.14 with ASE 3.29, pymatgen 2026.10.16 and NumPy 2.5.

## Quick table

| You see | What goes wrong | Fix |
|---|---|---|
| `Vasprun(path, exception_on_bad_xml=False)`, then `v.converged` | a run killed by the walltime reports `converged=True` | keep the default; a parse error means a failed run |
| `opt.run(fmax=..., steps=...)` with the result ignored | an unconverged structure is stored as relaxed | check the returned bool |
| `read("vasprun.xml").get_potential_energy()` as a label | the sigma→0 energy, and only for the last ionic step | `force_consistent=True` for the free energy; `index=":"` |
| `atoms.get_forces()` with `FixAtoms` (also after reading an OUTCAR, or an extxyz with `move_mask`) | fixed atoms get zero force | `get_forces(apply_constraint=False)` |
| `atoms.info["energy"]` after reading an extxyz | `KeyError` in ASE ≥3.23: the label is in `atoms.calc` | keys such as `REF_energy`; copy labels out before attaching a model |
| per-site lists indexed after `MPRelaxSet` or ASE's VASP calculator | sites come back sorted by species | per-site data on the structure (`site_properties`, `atoms.arrays`) |
| `Langevin(atoms, 1.0, temperature=300, friction=0.01)` | 10 fs steps at 3.5 million K | `1 * units.fs`, `temperature_K=300`, `0.01 / units.fs` |
| no `Stationary(atoms)` before NVE; `T = 2 * ke / (3 * N * kB)` | centre-of-mass drift swamps the MSD; T is wrong with constraints | `Stationary(atoms)`; `atoms.get_temperature()` |
| transport from a run with strong Langevin friction or per-step rescaling | D measures the thermostat | NVE, or weak global coupling, for production |
| MSD from `get_scaled_positions()` or LAMMPS `x y z` columns | wrapped coordinates: the MSD levels off | `xu yu zu`, or unwrap |
| `np.mean(d**2, axis=(1, 2))`; `intercept, slope = np.polyfit(...)` | D off by 3, by the dump interval, or meaningless | sum over x, y and z; check the time axis and the order |
| `x.std() / np.sqrt(len(x))` on a time series | error bars too small by √τ_int (14× in the example) | cut equilibration, then block averages |
| `squeue` as a completion check; `SLURM_ARRAY_TASK_ID` as a list index | failed jobs look finished; inputs off by one | `sacct`; a manifest written at submission |
| a checkpoint in `finally`; the same `trajectory=` on resubmission | SIGTERM skips `finally`; the old trajectory is deleted | a SIGTERM handler; `append_trajectory=True` |

## Examples

### Killed or unconverged runs

```python
v = Vasprun("vasprun.xml", exception_on_bad_xml=False)   # killed in ionic step 15 of 29
v.converged, v.final_energy        # (True, -269.36273275)
# UserWarning: XML is malformed. Parsing has stopped but partial data is available.
opt = BFGS(atoms)
opt.run(fmax=0.01, steps=10)       # False
relaxed_energy = atoms.get_potential_energy()
```

`converged_ionic` only checks that fewer than NSW ionic steps ran (14, NSW = 99), which any killed
relaxation satisfies, and ASE's `read("vasprun.xml")` returns the same energy without a warning.
`Optimizer.run` returns whether it converged; `steps` defaults to 100,000,000. **Fix:** keep
`exception_on_bad_xml=True` (this file then raises `XMLSyntaxError`) and act on `run()`'s value.

### Which energy, and which frames

```python
atoms = read("vasprun.xml")                         # a relaxation, ISMEAR = 1, SIGMA = 0.2
atoms.get_potential_energy()                        # -11.18981538: energy(sigma->0)
atoms.get_potential_energy(force_consistent=True)   # -11.217323: free energy (TOTEN)
Vasprun("vasprun.xml").final_energy                 # -11.18981538 eV: also sigma->0
len(read("vasprun.xml", index=":"))                 # 8: read() alone kept only the last step
```

ASE's docstring calls the free energy "the energy consistent with the forces"; here the two
differ by 27.5 meV/atom, and a dataset built from all three calls mixes conventions. **Fix:** one
convention for labels (the free energy when forces are fitted too), and `index=":"`.

### Labels: fixed atoms and extxyz keys

```python
slab.set_constraint(FixAtoms(indices=[0, 1, 2, 3]))      # Cu(111) slab, bottom layer fixed
slab.calc = EMT()
abs(slab.get_forces()[:4]).max()                         # 0.0
abs(slab.get_forces(apply_constraint=False)[:4]).max()   # 0.119
write("train.extxyz", slab)
abs(read("train.extxyz").get_forces()[:4]).max()         # 0.0: move_mask restores the constraint
frames = read("labels.extxyz", index=":")                # comment line has energy=-1.5
frames[0].info["energy"]                                 # KeyError: 'energy'
frames[0].get_potential_energy()                         # -1.5, from a SinglePointCalculator
frames[0].calc = EMT()                                   # stands in for the model under test
frames[0].get_potential_energy()                         # 1.1467: the label is gone
```

A loader that calls `get_forces()` trains on zeros, and ASE's OUTCAR reader loads constraints
from a POSCAR beside it (a VASP 5.2 OUTCAR: 0.305 eV/Å alone, 0.0 with a selective-dynamics
POSCAR). From ASE 3.23 extxyz `energy` and `forces` live only in a `SinglePointCalculator`
(3.22.1 also filled `info["energy"]`), which `copy()` drops and any atom move, even `wrap()`,
invalidates (`PropertyNotImplementedError`). **Fix:** `apply_constraint=False`; `REF_energy` keys.

### Input sets reorder atoms

```python
s = Structure(Lattice.cubic(10.0), ["Zn", "O", "C", "O", "O"], frac)   # framework, then CO2
vis = MPRelaxSet(s)                               # sort_structure=True by default
[str(sp) for sp in vis.structure.species]         # ['Zn', 'C', 'O', 'O', 'O']
co2 = vis.structure[2:]                           # "the last three sites": O, O, O
nacl.repeat((2, 1, 1)).get_chemical_symbols()     # ['Na', 'Cl', 'Na', 'Cl']
AseAtomsAdaptor.get_structure(nacl) * (2, 1, 1)   # sites: Na, Na, Cl, Cl
```

Inputs and outputs of this set are in sorted order, so per-site charges, tags or "the adsorbate
is the last three atoms" point at framework atoms. ASE's VASP calculator also sorts by species
and maps its own results back (`ase-sort.dat`), but raw files read later stay sorted. **Fix:**
per-site data on the structure (`site_properties`, `atoms.arrays`), or `sort_structure=False`.

### ASE MD units

```python
dyn = Langevin(atoms, timestep=1.0, temperature=300, friction=0.01)
dyn.dt / units.fs      # 10.18: the timestep is in ASE time units
dyn.temp / units.kB    # 3481356.6: `temperature` is in eV, with only a FutureWarning
dyn.fr * units.fs      # 0.00098 per fs: friction is per ASE time unit
```

`NVTBerendsen(temperature=300)` is in kelvin, so the keyword changes units between classes.
`MaxwellBoltzmannDistribution(atoms, 300)` set about 3 million K in ASE 3.28 (FutureWarning only)
and raises `TypeError` in 3.29; `pressure_au=1.0` is 160 GPa. **Fix:** `Langevin(atoms,
1 * units.fs, temperature_K=300, friction=0.01 / units.fs)`, and `* units.bar` for pressures.

### Centre-of-mass drift and degrees of freedom

```python
MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(1))  # 100 Ar
v_cm = atoms.get_momenta().sum(0) / atoms.get_masses().sum()
np.linalg.norm(v_cm) * units.fs * 1e5       # 83.9 m/s (43 m/s rms over draws)
gas.get_number_of_degrees_of_freedom()      # 50 for ten rigid CO2 (FixLinearTriatomic), not 90
T = 2 * gas.get_kinetic_energy() / (3 * len(gas) * units.kB)  # 0.556 × gas.get_temperature()
```

Velocity Verlet conserves momentum, so in NVE the box keeps moving (83.9 m/s after 200 steps):
that adds about 7,000 Å² to the MSD after 100 ps, against 6Dt = 120 Å² for D = 2 × 10⁻⁹ m²/s.
A thermostat holding this `T` at 300 K runs the rigid gas at 540 K. **Fix:** `Stationary(atoms)`
after drawing velocities, and `atoms.get_temperature()`, which subtracts constrained DOF.

### Thermostats and transport properties

```python
dyn = Langevin(gas, 2 * units.fs, temperature_K=300, friction=0.01 / units.fs, fixcm=False)
dyn.run(10_000)                       # then D from the MSD
# 200 non-interacting Ar atoms: D = 0.605 Å²/ps; with friction=0.001 / units.fs, D = 6.19 Å²/ps
```

For free particles Langevin dynamics gives D = kT/(mγ) (0.624 and 6.24 Å²/ps): the result reports
the friction you chose. With interactions the effect is smaller but has the same sign (Basconi and
Shirts, JCTC 2013); per-step rescaling and Berendsen break equipartition (Braun et al., JCTC 2018).
**Fix:** equilibrate with any thermostat, then measure in NVE or with weak global coupling (CSVR).

### MSD and the diffusion coefficient

```python
frames = read("dump.lammpstrj", index=":")   # columns: id type x y z xu yu zu
[f.positions[1, 0] for f in frames]           # [9.5, 0.5]: wrapped x; xu was 9.5, then 30.5
d = pos - pos[0]                              # pos: unwrapped, shape (frames, atoms, 3)
msd = np.mean(d**2, axis=(1, 2))              # D = 0.163: also averages over x, y and z
t = np.arange(len(pos)) * timestep            # D = 48.9: frames were written every 100 steps
intercept, slope = np.polyfit(t, msd, 1)      # 0.240: polyfit returns [slope, intercept]
D = slope / 6
```

ASE's LAMMPS reader prefers `x y z` to `xu yu zu`, and `get_scaled_positions()` wraps by default.
On a random walk in a 20 Å box (D = 0.5 Å²/ps), wrapped positions level the MSD off near
L²/2 = 200 Å² and give D = 0.066 instead of 0.451. The other comments are single slips on a walk
whose correct estimate is 0.489; also skip the ballistic start, and divide by 2d (2 in 1D pores).
**Fix:** unwrap, sum over x, y, z, use the dump interval, `slope, intercept = ...` (Maginn 2019).

### Error bars from correlated samples

```python
sem = loading.std(ddof=1) / np.sqrt(len(loading))   # 0.016
blocks = loading.reshape(50, -1).mean(axis=1)       # 50 blocks of 4,000 samples
blocks.std(ddof=1) / np.sqrt(len(blocks))           # 0.209
```

`loading` is a synthetic correlated series (AR(1), φ = 0.99, 200,000 samples) whose true
standard error is 0.226, 14× the naive one. Check that the block estimate levels off as blocks
grow, and cut equilibration (GCMC initialisation cycles, the start of an MD run) first. **Fix:**
pymbar's `timeseries.detect_equilibration`, then block averages (Grossfield et al., LiveCoMS 2018).

### Slurm arrays and job state

```python
task = int(os.environ["SLURM_ARRAY_TASK_ID"])   # sbatch --array=1-500
cif = sorted(glob.glob("cifs/*.cif"))[task]      # task 500: IndexError: list index out of range
done = subprocess.run(["squeue", "-h", f"--jobs={job_id}"],
                      capture_output=True, text=True).stdout == ""
```

`cifs[0]` never runs, and a listing taken at run time shifts if `cifs/` changes. A batch script
whose `python relax.py` crashes but whose last line (`cp -r out ...`) succeeds exits with 0 (plain
bash here); per Slurm's docs, that status sets the job state (COMPLETED), `squeue` lists only
pending, running and completing jobs, and `MaxArraySize` defaults to 1001. **Fix:** a manifest
written at submission, `sacct -X -n -P -o State,ExitCode`, `set -euo pipefail` (exit status 1).

### Walltime, SIGTERM and restarts

```python
try:
    dyn = Langevin(atoms, 1 * units.fs, temperature_K=300, friction=0.01 / units.fs,
                   trajectory="md.traj")   # resubmitted job: md.traj held 500 frames
    dyn.run(n_total)                        # md.traj was deleted; dyn.nsteps starts at 0
finally:
    save_checkpoint(atoms)                  # skipped when the process gets SIGTERM
```

Slurm sends SIGTERM at the time limit and SIGKILL `KillWait` seconds later (30 by default, per
its docs). Without a handler, a Python process killed by SIGTERM (return code -15) runs neither
`finally` nor `atexit`; with one that raises `SystemExit`, both run. **Fix:** that handler,
`#SBATCH --signal`, `append_trajectory=True` (100 frames became 111 after 10 more steps: the first
is written twice) and `dyn.run(n_total - n_done)`.

## Looks wrong, is fine

- `friction=0.01 / units.fs` divides because friction is a rate, and `NVTBerendsen(...,
  temperature=300)` really is kelvin, unlike Langevin's. Version-dependent: checked on ASE 3.29.
- `CifParser(path).parse_structures(primitive=False)` and `Structure.from_file(path)` keep the
  CIF's cell (4 sites for conventional fcc Cu); the deprecated `get_structures()` still reduces to
  the primitive cell (1 site, FutureWarning). Version-dependent: the switch came in 2023.11.10.
- `Vasprun.final_energy` and ASE's reader both give -269.38319884 eV for a VASP 5.2.2 file whose
  `<energy>` block says `e_0_energy` is 0.0; they rebuild it from the last SCF step.
- Unwrapping by summing minimum-image steps between frames looks ad hoc but is exact (it gave
  0.451 back above), as long as no atom moves more than half a cell between saved frames.
- `comp["Fe"]` on a `Composition` of Fe2+ and O2- returns 1, because element lookups sum over
  oxidation states. Version-dependent: pymatgen 2023.7.17 returned 0 (checked under Python 3.11).
