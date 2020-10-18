#!/bin/bash -l
#SBATCH --job-name="dns"
#SBATCH --time=24:00:00
#SBATCH --array=0-16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

BASE=/scratch/snx3000/novatig/CubismUP3D/
res=(60 65 70 76 82 88 95 103 111 120 130 140 151 163 176 190 205)
RE=${res[$SLURM_ARRAY_TASK_ID]}
srun ./sf_compute.py --simList ${BASE}/HITDNS03RK_UW_CFL010_BPD32_EXT2pi --Re $RE --recompute

