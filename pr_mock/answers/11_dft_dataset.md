# Answer key: 11_dft_dataset.py

**Good clarifying questions:** Do all runs, including the reference runs for the empty MOFs and
for CO2, use the same INCAR settings, k-points and dispersion correction? Is the framework fixed
in every run? What happens to runs killed at the walltime: are they resubmitted, and into the
same directory? Who uses the tags, and how much weight do they carry?

## Blockers

1. **Runs killed half-way pass the "finished" check** (`finished`). With
   `exception_on_bad_xml=False`, pymatgen parses as much of the file as it can. For a
   relaxation, `converged` only asks whether the run stopped before NSW and whether the last
   SCF converged, and a truncated file looks exactly like that. A copy of a 29-step relaxation
   cut during step 6 parsed 5 steps and reported `converged=True`. Its final energy was 0.098 eV
   above the finished run's, and its forces reached 0.85 eV/Å (the finished run ended at 0.105).
   Those energies go into `relaxed.csv` as relaxed binding energies, with errors of 0.03–0.1 eV
   on values of a few tenths of an eV. *Fix:* let a truncated XML raise; also require the final
   forces to be below the relaxation criterion; and log why each run was skipped.

2. **Fixed atoms get zero forces** (`labelled_frames`). ASE's OUTCAR reader picks up
   selective-dynamics constraints from a POSCAR or CONTCAR in the same directory, and
   `get_forces()` applies constraints by default, which zeroes the fixed atoms. In the check,
   the OUTCAR alone gave no constraints. With its POSCAR next to it, 2 of 7 atoms were fixed,
   and their `REF_forces` were exactly 0 where VASP printed up to 0.043 eV/Å. In these runs the
   whole framework is fixed, so the MLIP learns that framework atoms feel no force. The labels
   also depend on which files happen to be in the directory. *Fix:*
   `atoms.get_forces(apply_constraint=False)`, and remove the constraints before writing.

3. **The CO2 tags land on the wrong atoms.** MPRelaxSet sorts the sites (here Mg, H, C, O) when
   it writes the POSCAR, so the CO2 molecules aren't at the end any more. In the check, the
   "last 3n atoms" were six oxygens; the real CO2 atoms are 2 carbons and 4 oxygens. The trainer
   up-weights the wrong atoms. *Fix:* carry the label through the workflow as a site property
   (pymatgen) or in `atoms.arrays`, rather than by position.

## Major

4. **Electronic convergence isn't checked for each ionic step.** A step whose SCF stopped at NELM
   still writes energies and forces, and they'd go into the training set with the converged
   ones. In this fixture the worst step took 49 of the 60 allowed. *Fix:* drop steps whose SCF
   reached NELM, or check the last energy change against EDIFF.
5. **Deduplication uses only the rounded energy, across every run.** Two different structures
   from different runs, at −1234.5671 and −1234.5669 eV, collapsed into one frame. *Fix:*
   deduplicate within a run, and by geometry (maximum displacement) as well as energy.
6. **The binding energies use reference energies from other runs.** E(MOF) and E(CO2) come from
   separate runs, and any difference in settings (ENCUT, k-points, PREC, dispersion correction)
   goes straight into E_bind. Check that the settings match, for example by comparing INCARs,
   and store them next to the energies.
7. **One broken directory stops the whole build.** A run that crashed before writing
   `vasprun.xml` or `OUTCAR` raises FileNotFoundError, possibly after hours of parsing, and
   nothing is written. *Fix:* catch errors per run, and write a summary of the skipped runs with
   reasons.

## Minor

8. **Each file is parsed in full, twice over.** `finished()` parses each `vasprun.xml`, which can
   run to hundreds of MB, to read two values, and the OUTCAR is then parsed again for the frames.
   Parse once.
9. **Which energy?** `get_potential_energy()` from the OUTCAR reader is the σ→0 energy, while the
   forces are derivatives of the free energy (TOTEN). With small smearing in insulating MOFs the
   difference is tiny, but say which convention the labels use (`force_consistent=True` gives
   the free energy).
10. **`E_CO2` is hard-coded, and `E_MOF` is read at import time.** Pass them in, so the script
    can be tested without the reference files.

## Red herrings (say why they're fine)

- **`read(OUTCAR, index=":")`** reads every ionic step, as intended. The default, `index=-1`,
  would give only the last one.
- **`REF_energy` and `REF_forces`** are the right keys since ASE 3.23, because they survive a
  write and read. `atoms.calc = None` stops a second copy from being written.
- **`Vasprun(...).final_energy`** is the right value. pymatgen rebuilds it, which avoids a bug in
  the XML of some VASP versions.
- **`sorted(glob(...))`** gives a deterministic order.
- **`parse_dos=False, parse_eigen=False`** only skips data we don't need.

## Model 1-minute summary
"Three things put wrong numbers into the dataset with no error. Runs killed at the walltime
pass the 'finished' check, because a truncated vasprun parses as converged, so unrelaxed
energies go into the binding-energy table, off by up to 0.1 eV. ASE picks up the
selective-dynamics constraints from the POSCAR in the run directory, and `get_forces()` zeroes
the fixed atoms, so the whole framework gets zero force labels. The CO2 tags assume the
molecules are the last atoms, but MPRelaxSet sorted the sites, so six oxygens get tagged
instead. After those: no per-step SCF check, deduplication on energy alone across runs,
reference energies from runs with possibly different settings, and one broken directory
stopping the whole build. I'd ask for a test that builds from a fixture directory with a
truncated run and a selective-dynamics POSCAR, and checks the forces and tags."
