#!/usr/bin/env python3

import re, argparse, numpy as np, glob, subprocess
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

bDoRK23 = False
bDoRK23 = True
bDoUpWind = False
bDoUpWind = True

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
    return base + "BPD%d_RE%03d_CS_DSM_RUN%d" % (nblocks, re, run)
  else:
    return base + "BPD%d_RE%03d_CS%.03f_RUN%d" % (nblocks, re, cs, run)

def getSettings(nu, eps, cs, nblocks):
    options = '-sgs SSM -cs %f -bpdx %d -bpdy %d -bpdz %d -CFL 0.1 ' \
              % (cs, nblocks, nblocks, nblocks)
    if bDoRK23: options = options + '-RungeKutta23 1 '
    if bDoUpWind: options = options + '-Advection3rdOrder 1 '
    tAnalysis = 10 * np.sqrt(nu / eps)
    tEnd = 1e4 * tAnalysis
    return options + '-extentx 6.283185307179586 -dump2D 0 -dump3D 0 ' \
       '-tdump 0 -BC_x periodic -BC_y periodic -BC_z periodic ' \
       '-spectralIC fromFile -initCond HITurbulence -tAnalysis %f ' \
       '-compute-dissipation 1 -nprocsx 1 -nprocsy 1 -nprocsz 1 ' \
       '-spectralForcing 1 -tend %f -keepMomentumConstant 1 ' \
       '-analysis HIT -nu %f -energyInjectionRate %f ' \
       % (tAnalysis, tEnd, nu, eps)

def launchEuler(tpath, nu, eps, re, cs, nblocks, run):
    scalname = "%s/scalars_RE%03d" % (tpath, re)
    logEname = "%s/spectrumLogE_RE%03d" % (tpath, re)
    iCovname = "%s/invCovLogE_RE%03d" % (tpath, re)
    sdtDevname = "%s/stdevLogE_RE%03d" % (tpath, re)
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
      % (runname, scalname, logEname, iCovname, sdtDevname, \
        runname, getSettings(nu, eps, cs, nblocks))
    subprocess.run(cmd, shell=True)


def launchDaint(nCases, les):
    SCRATCH = os.getenv('SCRATCH')
    HOME = os.getenv('HOME')

    f = open('HIT_sbatch','w')
    f.write('#!/bin/bash -l \n')
    if les:
      f.write('#SBATCH --job-name=LES_HIT \n')
      f.write('#SBATCH --time=01:00:00 \n')
    else:
      f.write('#SBATCH --job-name=DNS_HIT \n')
      f.write('#SBATCH --time=24:00:00 \n')
    f.write('#SBATCH --output=out.%j.%a.txt \n')
    f.write('#SBATCH --error=err.%j.%a.txt \n')
    f.write('#SBATCH --constraint=gpu \n')
    f.write('#SBATCH --account=s929 \n')
    f.write('#SBATCH --array=0-%d \n' % (nCases-1))
    #f.write('#SBATCH --partition=normal \n')
    #f.write('#SBATCH --ntasks-per-node=1 \n')

    f.write('ind=$SLURM_ARRAY_TASK_ID \n')
    if les:
      f.write('RUNDIRN=`./launchLESHIT.py --LES --case ${ind} --printName` \n')
      f.write('OPTIONS=`./launchLESHIT.py --LES --case ${ind} --printOptions` \n')
    else:
      f.write('RUNDIRN=`./launchLESHIT.py --case ${ind} --printName` \n')
      f.write('OPTIONS=`./launchLESHIT.py --case ${ind} --printOptions` \n')

    f.write('mkdir -p %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cd %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cp %s/CubismUP_3D/bin/simulation ./exec \n' % HOME)

    f.write('export OMP_NUM_THREADS=12 \n')
    f.write('export CRAY_CUDA_MPS=1 \n')
    f.write('srun --ntasks 1 --ntasks-per-node=1 ./exec ${OPTIONS} \n')
    f.close()
    os.system('sbatch HIT_sbatch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")

    parser.add_argument('--path', default='target', help="Simulation case.")
    parser.add_argument('--nBlocksRL', type=int, default=4,
    help="Number of CubismUP 3D blocks in the training runs.")
    args = parser.parse_args()

    #for re in [60, 70, 82, 95, 111, 130, 152, 176]:
    for ri in [2, 3]:
      #for cs in np.linspace(0.26, 0.4, 8):
      #for cs in [-0.02]:
      #for cs in [-.01, 0, .01, .02, .03, .04, .05, .06, .07, .1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2, .21, .22, .23, .24, .25, .26, .27, .28, .29, .3, .31, .32]:
      for cs in np.linspace(-0.01, 0.32, 34):
        for re in [60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205] :
          eps, nu = epsNuFromRe(re)
          launchEuler(args.path, nu, eps, re, cs, args.nBlocksRL, ri)

