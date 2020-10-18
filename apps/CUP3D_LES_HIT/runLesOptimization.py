#!/usr/bin/env python3

import re, argparse, numpy as np, glob, subprocess, os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bDoRK23 = True
bDoUpWind = False
bDoUpWind = True

runsname = "HITLES_GRID"
CSs = [0.2, 0.2075, 0.215, 0.22, 0.225, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23]
REs = [60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205]

def epsNuFromRe(Re, uEta = 1.0):
    C = 3.0 # np.sqrt(196.0/20.0) #2.87657077
    K = 2/3.0 * C * np.sqrt(15)
    eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps
    return eps, nu

def runspec(re, cs, nblocks, run):
  if bDoRK23 : tstep = "RK_"
  else : tstep = "FE_"
  if bDoUpWind : discr = "UW_"
  else : discr = "CD_"
  base = "HITLES00_" + tstep + discr + "CFL010_"
  if cs < 0:
    return base + "BPD%d_CS_DSM_RUN%d_RE%03d" % (nblocks, run, re)
  else:
    return base + "BPD%d_CS%.03f_RUN%d_RE%03d" % (nblocks, cs, run, re)

def getSettings(nu, eps, cs, nblocks):
    options = '-sgs SSM -cs %f -bpdx %d -bpdy %d -bpdz %d -CFL 0.1 ' \
              % (cs, nblocks, nblocks, nblocks)
    if bDoRK23:   options = options + '-RungeKutta23 1 '
    else:         options = options + '-RungeKutta23 0 '
    if bDoUpWind: options = options + '-Advection3rdOrder 1 '
    else:         options = options + '-Advection3rdOrder 0 '
    tAnalysis = np.sqrt(nu / eps)
    tEnd = 1e4 * tAnalysis # abort in code after N t_integral
    return options + '-extentx 6.283185307179586 -dump2D 0 -dump3D 1 ' \
       '-tdump %f -BC_x periodic -BC_y periodic -BC_z periodic ' \
       '-spectralIC fromFile -initCond HITurbulence -tAnalysis %f ' \
       '-compute-dissipation 1 -nprocsx 1 -nprocsy 1 -nprocsz 1 ' \
       '-spectralForcing 1 -tend %f -keepMomentumConstant 1 ' \
       '-analysis HIT -nu %f -energyInjectionRate %f ' \
       % (0*tAnalysis, tAnalysis, tEnd, nu, eps)

def launchEuler(tpath, nu, eps, re, cs, nblocks, run):
    scalname = "%s/scalars_RE%03d" % (tpath, re)
    logEname = "%s/spectrumLogE_RE%03d" % (tpath, re)
    iCovname = "%s/invCovLogE_RE%03d" % (tpath, re)
    sdtDevname = "%s/stdevLogE_RE%03d" % (tpath, re)
    settings = getSettings(nu, eps, cs, nblocks)
    runname  = runspec(re, cs, nblocks, run)

    cmd = "export LD_LIBRARY_PATH=/cluster/home/novatig/hdf5-1.10.1/gcc_6.3.0_openmpi_2.1/lib/:$LD_LIBRARY_PATH\n" \
      "FOLDER=/cluster/scratch/novatig/CubismUP3D/%s\n " \
      "mkdir -p ${FOLDER}\n" \
      "cp ~/CubismUP_3D/bin/simulation ${FOLDER}/\n" \
      "cp %s ${FOLDER}/scalars_target\n" \
      "cp %s ${FOLDER}/spectrumLogE_target\n" \
      "cp %s ${FOLDER}/invCovLogE_target\n" \
      "cp %s ${FOLDER}/stdevLogE_target\n" \
      "export OMP_NUM_THREADS=8\n" \
      "cd $FOLDER\n" \
      "bsub -n 8 -J %s -W 04:00 -R \"select[model==XeonGold_6150] span[ptile=8]\" mpirun -n 1 ./simulation %s\n" \
      % (runname, scalname, logEname, iCovname, sdtDevname, runname, settings)
    subprocess.run(cmd, shell=True)


def launchDaint(tpath, nu, eps, re, cs, nblocks, run):
    SCRATCH, HOME = os.getenv('SCRATCH'), os.getenv('HOME')
    appdir = "%s/smarties/apps/CUP3D_LES_HIT" % HOME
    scalname   = "%s/%s/scalars_RE%03d"      % (appdir, tpath, re)
    logEname   = "%s/%s/spectrumLogE_RE%03d" % (appdir, tpath, re)
    iCovname   = "%s/%s/invCovLogE_RE%03d"   % (appdir, tpath, re)
    sdtDevname = "%s/%s/stdevLogE_RE%03d"    % (appdir, tpath, re)
    settings = getSettings(nu, eps, cs, nblocks)
    runname  = runspec(re, cs, nblocks, run)

    f = open('HIT_sbatch','w')
    f.write('#!/bin/bash -l \n')
    f.write('#SBATCH --job-name=LES_HIT \n')
    f.write('#SBATCH --time=01:00:00 \n')
    #f.write('#SBATCH --time=00:30:00 \n')
    f.write('#SBATCH --output=out.%j.%a.txt \n')
    f.write('#SBATCH --error=err.%j.%a.txt \n')
    f.write('#SBATCH --constraint=gpu \n')
    f.write('#SBATCH --account=s929 \n')

    f.write('mkdir -p %s/CubismUP3D/%s \n' % (SCRATCH, runname) )
    f.write('cd       %s/CubismUP3D/%s \n' % (SCRATCH, runname) )
    f.write('cp %s/CubismUP_3D/bin/simulation ./exec \n' % HOME)
    f.write('cp %s ./scalars_target \n'      % scalname)
    f.write('cp %s ./spectrumLogE_target \n' % logEname)
    f.write('cp %s ./invCovLogE_target \n'   % iCovname)
    f.write('cp %s ./stdevLogE_target \n'    % sdtDevname)
    f.write('export OMP_NUM_THREADS=12 \n')
    f.write('export CRAY_CUDA_MPS=1 \n')
    f.write('export MPICH_MAX_THREAD_SAFETY=multiple \n')
    f.write('srun --ntasks 1 --ntasks-per-node=1 ./exec %s\n' % settings)
    f.close()
    os.system('sbatch HIT_sbatch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")

    parser.add_argument('--path', default='target_RK_2blocks', help="Simulation case.")
    parser.add_argument('--nBlocks', type=int, default=2,
    help="Number of CubismUP 3D blocks in the training runs.")
    args = parser.parse_args()

    for ri in [4]:
      #CSs = [-0.1] * len(REs)
      for re, cs in zip(REs, CSs):
        eps, nu = epsNuFromRe(re)
        launchDaint(args.path, nu, eps, re, cs, args.nBlocks, ri)
        #launchEuler(args.path, nu, eps, re, cs, args.nBlocks, ri)

