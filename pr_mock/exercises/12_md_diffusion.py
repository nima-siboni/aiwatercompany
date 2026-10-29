"""
PR #268: CO2 self-diffusion in the top-50 MOFs from MLIP molecular dynamics

For each of the 50 best MOFs from the GCMC screen, runs 2 ns of Langevin MD at 300 K on the
framework loaded with CO2 (starting from the GCMC snapshots, guest atoms tagged 1), using our
fine-tuned MACE potential. Then fits the CO2 self-diffusion coefficient from the mean-squared
displacement of the CO2 carbons. One Slurm array task per MOF:
`sbatch --array=1-50 --time=48:00:00 run_md.sh`. Tasks that hit the time limit are resubmitted
and continue from their trajectory.
"""
import glob
import os

import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mlip import load_calculator  # fine-tuned MACE, as an ASE calculator (float32)

TEMPERATURE_K = 300
TIMESTEP = 1.0  # fs
N_STEPS = 2_000_000  # 2 ns
INTERVAL = 100  # write a frame every 100 steps


def run_md(structure_path, traj_path):
    if os.path.exists(traj_path):
        atoms = read(traj_path, index=-1)  # resume from the last frame
    else:
        atoms = read(structure_path)
    atoms.calc = load_calculator()
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K)
    dyn = Langevin(atoms, timestep=TIMESTEP, temperature_K=TEMPERATURE_K,
                   friction=0.01 / units.fs, trajectory=traj_path, loginterval=INTERVAL)
    dyn.attach(atoms.wrap, interval=INTERVAL)  # keep molecules inside the cell for visualisation
    dyn.run(N_STEPS)


def co2_carbon_positions(traj_path):
    frames = read(traj_path, index=":")
    carbons = [a.index for a in frames[0] if a.symbol == "C" and a.tag == 1]
    return np.array([f.positions[carbons] for f in frames])


def diffusion_coefficient(pos, dt_fs=TIMESTEP, interval=INTERVAL):
    """pos: (n_frames, n_molecules, 3) in Å. Returns D and its standard error, in cm^2/s."""
    msd = ((pos - pos[0]) ** 2).mean(axis=(1, 2))
    t_fs = np.arange(len(msd)) * dt_fs * interval
    (slope, _), cov = np.polyfit(t_fs, msd, 1, cov=True)
    to_cm2_s = 1e-16 / 1e-15  # Å^2/fs -> cm^2/s
    return slope / 6 * to_cm2_s, np.sqrt(cov[0, 0]) / 6 * to_cm2_s


def main():
    mofs = sorted(glob.glob("top50/*_co2.extxyz"))
    mof = mofs[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    name = os.path.basename(mof).removesuffix("_co2.extxyz")
    traj = f"md/{name}.traj"
    try:
        run_md(mof, traj)
    finally:
        d, err = diffusion_coefficient(co2_carbon_positions(traj))
        with open(f"md/{name}_D.txt", "w") as f:
            f.write(f"{d:.3e} {err:.1e}\n")


if __name__ == "__main__":
    main()
