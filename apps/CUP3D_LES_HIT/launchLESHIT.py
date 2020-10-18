#!/usr/bin/env python3
import os, numpy as np, argparse, subprocess

bDoRK23 = False
bDoRK23 = True
bDoUpWind = False
bDoUpWind = True
BPD=32
base = "HITDNS03"
NNODES=32
RUNSIDS = range(1)
ALLRES=[60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205]
ALLRES=[60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190]
#ALLRES=[205]

def epsNuFromRe(Re, uEta = 1.0):
    C = 3.0 # np.sqrt(20.0/3)
    K = 2/3.0 * C * np.sqrt(15)
    eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps
    return eps, nu

def runspec(nu, eps, re, run, cs):
    if bDoRK23 : tstep = "RK_"
    else : tstep = "FE_"
    if bDoUpWind : discr = "UW_"
    else : discr = "CD_"
    size = "CFL010_BPD%d_EXT2pi_" % (BPD)
    return base + tstep + discr + size + "RE%04d_RUN%d" % (re, run)

def getSettings(nu, eps, cs, run):
    options = '-bpdx %d -bpdy %d -bpdz %d -CFL 0.1 ' % (BPD, BPD, BPD)
    if bDoRK23: options = options + '-RungeKutta23 1 '
    if bDoUpWind: options = options + '-Advection3rdOrder 1 '
    tAnalysis = np.sqrt(nu / eps)
    tDump = 10 * tAnalysis
    tEnd = 10000 * tAnalysis # abort in code
    options = options + '-nprocsx %d -nprocsy 1 -nprocsz 1 ' % NNODES
    return options + '-extentx 6.2831853072 -dump2D 0 -dump3D 1 ' \
       '-tdump %f -BC_x periodic -BC_y periodic -BC_z periodic ' \
       '-spectralIC fromFit -initCond HITurbulence -spectralForcing 1 ' \
       '-compute-dissipation 1 -tAnalysis %f -keepMomentumConstant 1 ' \
       '-tend %f -analysis HIT -nu %f -energyInjectionRate %f ' \
       % (tDump, tAnalysis, tEnd, nu, eps)

def launchEuler(nu, eps, re, cs, run):
    runname  = runspec(nu, eps, re, run, cs)
    print(runname)
    cmd = "export LD_LIBRARY_PATH=/cluster/home/novatig/hdf5-1.10.1/gnu630_ompi30/lib/:$LD_LIBRARY_PATH\n" \
      "FOLDER=/cluster/scratch/novatig/CubismUP_3D/%s\n " \
      "mkdir -p ${FOLDER}\n" \
      "cp ~/CubismUP_3D/bin/simulation ${FOLDER}/\n" \
      "export OMP_NUM_THREADS=18\n" \
      "cd $FOLDER\n" \
      "bsub -n 18 -J %s -W 24:00 -R \"select[model==XeonGold_6150] span[ptile=18]\" ./simulation %s\n" \
      % (runname, runname, getSettings(nu, eps, cs, run))
      #"bsub -n 18 -J %s -W 24:00 -R \"select[model==XeonGold_6150] span[ptile=18]\" mpirun -n 1 ./simulation %s\n" \
    subprocess.run(cmd, shell=True)

def launchDaint(nCases, les):
    SCRATCH = os.getenv('SCRATCH')
    HOME = os.getenv('HOME')

    f = open('HIT_sbatch','w')
    f.write('#!/bin/bash -l \n')
    f.write('#SBATCH --job-name=DNS_HIT \n')
    f.write('#SBATCH --time=24:00:00 \n')
    f.write('#SBATCH --output=out.%j.%a.txt \n')
    f.write('#SBATCH --error=err.%j.%a.txt \n')
    f.write('#SBATCH --constraint=gpu \n')
    f.write('#SBATCH --account=s929 \n')
    f.write('#SBATCH --nodes=%d \n' % NNODES)
    f.write('#SBATCH --array=0-%d \n' % (nCases-1))
    #f.write('#SBATCH --partition=normal \n')
    #f.write('#SBATCH --ntasks-per-node=1 \n')
    
    f.write('ind=$SLURM_ARRAY_TASK_ID \n')
    #f.write('RUNDIRN=`./launchLESHIT.py --LES --case ${ind} --printName` \n')
    #f.write('OPTIONS=`./launchLESHIT.py --LES --case ${ind} --printOptions` \n')
    f.write('RUNDIRN=`./launchLESHIT.py --case ${ind} --printName` \n')
    f.write('OPTIONS=`./launchLESHIT.py --case ${ind} --printOptions` \n')

    f.write('mkdir -p %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cd %s/CubismUP3D/${RUNDIRN} \n' % SCRATCH)
    f.write('cp %s/CubismUP_3D/bin/simulation ./exec \n' % HOME)

    f.write('export MPICH_MAX_THREAD_SAFETY=multiple \n')
    f.write('export OMP_NUM_THREADS=12 \n')
    f.write('export CRAY_CUDA_MPS=1 \n')
    #f.write('srun --ntasks 1 --ntasks-per-node=1 ./exec ${OPTIONS}\n')
    f.write('srun --nodes %d --ntasks-per-node=1 ./exec ${OPTIONS}\n' % NNODES)
    f.close()
    os.system('sbatch HIT_sbatch')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")

    parser.add_argument('--printName', dest='printName',
      action='store_true', help="Only print run name.")
    parser.set_defaults(printName=False)

    parser.add_argument('--printOptions', dest='printOptions',
      action='store_true', help="Only print run options.")
    parser.set_defaults(printOptions=False)

    parser.add_argument('--launchDaint', dest='launchDaint',
      action='store_true', help="Only print run options.")
    parser.set_defaults(launchDaint=False)

    parser.add_argument('--launchEuler', dest='launchEuler',
      action='store_true', help="Only print run options.")
    parser.set_defaults(launchEuler=False)

    parser.add_argument('--LES', dest='LES', action='store_true',
      help="Triggers LES modeling.")
    parser.set_defaults(LES=False)

    parser.add_argument('--case', type = int, default = -1,
      help="Simulation case.")

    args = parser.parse_args()
    if args.LES: rangeles = np.linspace(0.16, 0.24, 9)
    else: rangeles = [None]

    NUS, EPS, RES, RUN, CSS = [], [], [], [], []

    for i in RUNSIDS :
      for les in rangeles :
        for re in ALLRES:
        #for re in [60, 70, 82, 95, 111, 130, 151, 176, 205] :
        #for re in [100] :
          RES, RUN = RES + [re], RUN + [i]
          eps, nu = epsNuFromRe(re, uEta = 1)
          NUS, EPS, CSS = NUS + [nu], EPS + [eps], CSS + [les]
          #if i==0: print( runspec(nu, eps, 0, None) )
    #exit()
    nCases = len(NUS)
    #print('Defined %d cases' % nCases)

    if args.launchDaint: launchDaint(nCases, args.LES)

    if args.case < 0: cases = range(nCases)
    else: cases = [args.case]

    for i in cases:
      if args.printOptions:
        print( getSettings(NUS[i], EPS[i], CSS[i], RUN[i]) )
      if args.printName:
        print( runspec(NUS[i], EPS[i], RES[i], RUN[i], CSS[i]) )
      if args.launchEuler:
           launchEuler(NUS[i], EPS[i], RES[i], CSS[i], RUN[i])


