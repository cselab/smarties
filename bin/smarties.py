#!/usr/bin/env python3
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
import argparse, os, psutil, sys, shutil

SCRATCH       = os.getenv('SCRATCH') or ''
SMARTIES_ROOT = os.getenv('SMARTIES_ROOT') or ''
HOSTNAME      = os.popen("hostname").read()

def isEuler():
  return HOSTNAME[:5]=='euler' or HOSTNAME[:3]=='eu-'

def isDaint():
  return HOSTNAME[:5]=='daint'

def is_exe(fpath):
  return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def getDefaults():
  if isEuler():
    print('Detected ETHZ cluster Euler')
    runprefix = SCRATCH + '/smarties/'
    nThreads = 36
  elif isDaint():
    print('Detected CSCS cluster Piz Daint')
    runprefix = SCRATCH + '/smarties/'
    nThreads = 12
  else:
    runprefix = SMARTIES_ROOT + '/runs/'
    nThreads = psutil.cpu_count(logical = False)
  return runprefix, nThreads

def copySettingsFiles(settings, absRunPath):
  for i in range(len(settings)):
    sett = settings[i]
    if os.path.isfile(sett) is False: # may omit path to smarties/settings/
      sett = SMARTIES_ROOT + '/settings/' + sett
    if os.path.isfile(sett) is False:
      print('FATAL: Did not find the specified settings file %s' % settings[i])
      exit()

    if len(settings) == 1: dest = absRunPath + '/settings.json'
    else : dest = absRunPath + '/settings_%02d.json' % i
    shutil.copy(sett, dest)

def applicationSetup(parsed, absRunPath):
  # will run application contained in the OpenAI gym library:
  if parsed.gymApp:
    assert(parsed.mpiProcsPerEnv <= 1)
    parsed.execname = "exec.py"
    gymScriptPath = SMARTIES_ROOT + '/apps/OpenAI_gym/'
    shutil.copy(gymScriptPath + "HumanoidWrapper.py", absRunPath + '/')
    shutil.copy(gymScriptPath + "exec.py", absRunPath + '/')
    parsed.args = parsed.app + parsed.args
    return
  # will run application contained in the OpenAI gym atari library:
  if parsed.atariApp:
    assert(parsed.mpiProcsPerEnv <= 1)
    parsed.execname = "exec.py"
    atariScriptPath = SMARTIES_ROOT + '/apps/OpenAI_gym_atari/'
    shutil.copy(atariScriptPath + "exec.py", absRunPath + '/')
    parsed.args = parsed.app + parsed.args
    return
  # will run application contained in the Deepmind control suite:
  if parsed.dmcApp:
    assert(parsed.mpiProcsPerEnv <= 1)
    parsed.execname = "exec.py"
    atariScriptPath = SMARTIES_ROOT + '/apps/Deepmind_control/'
    shutil.copy(atariScriptPath + "exec.py", absRunPath + '/')
    parsed.args = parsed.app + parsed.args
    os.system('export DISABLE_MUJOCO_RENDERING=1')
    return

  # Else user created app. First find its folder.
  if os.path.isdir( parsed.app ) :
    app = parsed.app
  elif os.path.isdir( SMARTIES_ROOT + '/apps/' + parsed.app ) :
    app = SMARTIES_ROOT + '/apps/' + parsed.app
  else:
    print('FATAL: Specified application directory %s not found.' % parsed.app)
    exit()

  # Now copy executable over to rundir, and if needed run a setup script:
  if is_exe(app + '/setup.sh'):
    setcmd = "cd %s && source ./setup.sh \n " \
             "echo ${EXTRA_LINE_ARGS}  \n " \
             "echo ${MPI_RANKS_PER_ENV} \n " \
             "echo ${EXECNAME}" % app
    setout = os.popen(setcmd).read()

    if setout.splitlines()[-3]:
      parsed.args = parsed.args + " " + setout.splitlines()[-3]

    if setout.splitlines()[-2]:
      mpiProcsPerEnv = int( setout.splitlines()[-2] )
      if parsed.mpiProcsPerEnv > 0:
        assert ( parsed.mpiProcsPerEnv == mpiProcsPerEnv ), \
               "Contradiction between application setup and cmd line parsing"
      parsed.mpiProcsPerEnv = mpiProcsPerEnv

    if setout.splitlines()[-1] and parsed.execname is not 'exec':
      parsed.execname = setout.splitlines()[-1]

  elif is_exe(app+'/'+parsed.execname):
    shutil.copy(app+'/'+parsed.execname, absRunPath + '/')
  elif is_exe(app+'/'+parsed.execname+'.py'):
    shutil.copy(app+'/'+parsed.execname+'.py', absRunPath + '/')
    parsed.execname = parsed.execname + '.py'

  elif is_exe(absRunPath+'/'+parsed.execname):
    print('WARNING: Using executable already located in run directory.')
  elif is_exe(absRunPath+'/'+parsed.execname+'.py'):
    print('WARNING: Using python executable already located in run directory.')
    parsed.execname = parsed.execname + '.py'

  else:
    print('FATAL: Unable to locate application executable')

def setComputationalResources(parsed):
  if parsed.mpiProcsPerEnv == 0: # 'forkable' applications
    # at least one learner process:
    if parsed.nProcesses < 1: parsed.nProcesses = max(1, parsed.nLearners)
    # surely now nProcesses is number of mpi processes
    if parsed.nLearners  < 1:
      # by default, maximize number of leaner processes, one on each available
      # rank. Use fork to spawn application env processes
      parsed.nLearners = parsed.nProcesses
    else: # nLearners was set, it could differ from nProcesses
      nWorkerProcesses = parsed.nProcesses - parsed.nLearners
      assert(nWorkerProcesses >= 0)
      if nWorkerProcesses > 0: # will use mpi worker processes:
        parsed.nEnvironments = max(parsed.nEnvironments, nWorkerProcesses)
    # if not requested mpi worker processes, have 1 forked env sim per learner:
    if parsed.nEnvironments == 0: parsed.nEnvironments = parsed.nLearners

  else:
    if parsed.nLearners < 1: parsed.nLearners = 1
    if parsed.nEnvironments < 1: parsed.nEnvironments = 1
    minNprocs = parsed.nLearners + parsed.nEnvironments * parsed.mpiProcsPerEnv
    if parsed.nProcesses < minNprocs: parsed.nProcesses = minNprocs

    # by default, maximize number of env processes:
    parsed.nEnvironments = (parsed.nProcesses - parsed.nLearners) / parsed.mpiProcsPerEnv

  parsed.args += " --nEnvironments %d --nMasters %d --nThreads %d" \
                 " --workerProcessesPerEnv %d " % (parsed.nEnvironments, \
                 parsed.nLearners, parsed.nThreads, parsed.mpiProcsPerEnv)

  if parsed.nEvalSeqs == 0:
    parsed.args += " --bTrain 1 --nTrainSteps %d " % parsed.nTrainSteps
    if parsed.restart is None: parsed.args += " --restart none "
    else: parsed.args += " --restart %s " % parsed.restart
  else:
    parsed.args += " --bTrain 0 --totNumSteps %d " % parsed.nEvalSeqs
    if parsed.restart is None: parsed.args += " --restart ./ "
    else: parsed.args += " --restart %s " % parsed.restart

  if parsed.netsOnlyLearners:
    parsed.args += " --learnersOnWorkers 0 "
  if parsed.printAppStdout:
    parsed.args += " --redirectAppStdoutToFile 0 "
  if parsed.disableDataLogging:
    parsed.args += " --logAllSamples 0 "


def setEnvironmentFlags(nThreads):
  return "unset LSB_AFFINITY_HOSTFILE  #euler cluster \n" \
         "export MPICH_MAX_THREAD_SAFETY=multiple #MPICH \n" \
         "export MV2_ENABLE_AFFINITY=0 #MVAPICH \n" \
         "export OMP_NUM_THREADS=%d \n" \
         "export OPENBLAS_NUM_THREADS=1 \n" \
         "export CRAY_CUDA_MPS=1 \n" \
         "export PYTHONPATH=${PYTHONPATH}:${SMARTIES_ROOT}/lib \n" \
         "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib \n" \
         "export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib \n " \
         % nThreads

def setLaunchCommand(parsed):
  nProcesses, nThreads, rundir = parsed.nProcesses, parsed.nThreads, parsed.runname
  clockHours = int(parsed.clockHours)
  clockMinutes = int((parsed.clockHours - clockHours) / 60)
  # default:
  #cmd = "mpirun -n %d --map-by ppr:%d:node ./%s %s | tee out.log" \
  #      % (nProcesses, parsed.nTaskPerNode, parsed.execname, parsed.args)
  cmd = "mpirun -n %d ./%s %s | tee out.log" \
        % (nProcesses, parsed.execname, parsed.args)

  if isEuler() and parsed.interactive is False:
    assert rundir is not None, "--runname option is required on Euler and Daint"
    if nThreads == 18:
      map_by = "--map-by ppr:2:node"
    elif nThreads == 36:
      map_by = "--map-by ppr:1:node"
    else:
      map_by = "--map-by ppr:%d:node" % parsed.nTaskPerNode
    cmd = "bsub -n %d -R \"select[model==XeonGold_6150] span[ptile=%d]\" " \
          " -J %s -W %s:00 mpirun -n %d %s ./%s %s " \
          % (nProcesses * nThreads, nThreads, rundir, clockHours, \
             nProcesses, map_by, parsed.execname, parsed.args)

  elif isDaint() and parsed.interactive is False:
    assert rundir is not None, "--runname option is required on Euler and Daint"
    sbatch = "#!/bin/bash -l \n" \
             "#SBATCH --account=s929 --time=%s:00:00 --job-name=%s \n" \
             "#SBATCH --output=%s_out_%%j.txt --error=%s_err_%%j.txt \n" \
             "#SBATCH --nodes=%d --constraint=gpu \n" \
             "srun -n %d --nodes=%d --ntasks-per-node=%d ./%s %s" \
             % (parsed.clockHours, rundir, rundir, rundir, nNodes, nProcs, \
                nNodes, parsed.nTaskPerNode, parsed.execname, parsed.args)
    cmd = "chmod 755 daint_sbatch \n sbatch daint_sbatch"

  elif isDaint() and parsed.interactive is True:
    cmd = "srun -n %d --nodes %d --ntasks-per-node%d ./%s %s" \
          % (nProcs, nNodes, nPerNode, parsed.execname, parsed.args)
  return cmd

if __name__ == '__main__':
  assert len(SMARTIES_ROOT)>0, \
         "FATAL: Environment variable SMARTIES_ROOT is unset. Read the README"
  assert os.path.isfile(SMARTIES_ROOT+'/lib/libsmarties.so'), \
         "FATAL: smarties library not found."

  runprefix, nThreads = getDefaults()

  parser = argparse.ArgumentParser(
      description = "Smarties launcher.")

  parser.add_argument('app', default='./', nargs='?',
      help='Specifier of the task to train on. This can be: ' \
           '    1) path to a directory containing the application executable. ' \
           '    2) name of a directory within SMARTIES_ROOT/apps/. ' \
           '    3) an environment of OpenAI gym if the --gym option is used. ' \
           '    4) an atari game (NoFrameskip-v4 will be added internally) ' \
           '       if the --atari option is used. ' \
           '    5) a Deepmind control suite env and task such as ' \
           '       \"acrobot swingup\", if the --dmc option is used. ' \
           '    The default value is \'./\' and smarties will look for a binary '
           '    or Python executable in the current directory.')

  parser.add_argument('settings', default=['VRACER.json'], nargs='*',
      help="(optional) path or name of the settings file specifying RL solver " \
           "and its hyper-parameters. The default setting file is set to VRACER.json")

  parser.add_argument('--runname', default=None,
      help="Name of the directory in which the learning process will be executed. " \
           "If unset, execution will take place in the current directory.")

  parser.add_argument('--nThreads', type=int, default=nThreads,
      help="(optional) Number of threads used by the learning processes. " \
           "The default value is the number of available CPU cores, here %d." \
           % nThreads)
  parser.add_argument('--nProcesses', type=int, default=0, # 0 tells me no expressed preference
      help="(optional) Number of processes available to run the training.")
  parser.add_argument('--nLearners', type=int, default=0, # 0 tells me no expressed preference
      help="(optional) Number of processes dedicated to update the networks. By default 1.")
  parser.add_argument('--nEnvironments', type=int, default=0, # 0 tells me no expressed preference
      help="(optional) Number of concurrent environment simulations. By default 1.")
  parser.add_argument('--mpiProcsPerEnv', type=int, default=0, # 0 tells me no expressed preference
    help="(optional) MPI processes required per env simulation. This value can also " \
         "be specified in app's setup.sh script by setting the MPI_RANKS_PER_ENV " \
         "shell variable. If unset or 0, smarties performs communication via " \
         "sockets and avoids creating multiple MPI processes.")

  parser.add_argument('--netsOnlyOnLearners', dest='netsOnlyLearners', action='store_true',
    help="(optional) Forces network approximator to live only inside learning " \
         "processes. If this option is used, workers send states to learners " \
         "and learners reply with actions. Otherwise, workers collect entire " \
         "episodes, send them to learners and receive parameter updates.")
  parser.set_defaults(netsOnlyLearners=False)

  parser.add_argument('--printAppStdout', dest='printAppStdout', action='store_true',
    help="(optional) Prints application output to screen. If unset, application " \
         "output will be redirected to file in the simulation subfolder.")
  parser.set_defaults(printAppStdout=False)

  parser.add_argument('--disableDataLogging', dest='disableDataLogging', action='store_true',
    help="(optional) Stops smarties from storing all state/action/reward/policy " \
         "into (binary) log files. These files enable postprocessing and analysis " \
         " but may occupy a lot of storage space.")
  parser.set_defaults(disableDataLogging=False)

  parser.add_argument('--restart', default=None,
      help="Path to existing directory which contains smarties output files "
           "needed to restart already trained agents.")
  parser.add_argument('--nTrainSteps', type=int, default=10000000,
      help="(optional) Total number of time steps before end of learning.")
  parser.add_argument('--nEvalSeqs', type=int, default=0,
      help="(optional) Number of environment episodes to evaluate trained policy. " \
           "This option automatically disables training.")

  parser.add_argument('--gym', dest='gymApp', action='store_true',
    help="(optional) Set if application is part of OpenAI gym.")
  parser.set_defaults(gymApp=False)
  parser.add_argument('--atari', dest='atariApp', action='store_true',
    help="(optional) Set if application is part of OpenAI gym's atari suite.")
  parser.set_defaults(atariApp=False)
  parser.add_argument('--dmc', dest='dmcApp', action='store_true',
    help="(optional) Set if application is part of DeepMind control suite.")
  parser.set_defaults(dmcApp=False)

  parser.add_argument('--execname', default='exec',
      help="(optional) Name of application's executable.")
  parser.add_argument('--runprefix', default=runprefix,
      help="(optional) Path to directory where run folder will be created.")

  parser.add_argument('--interactive', dest='interactive', action='store_true',
    help="(optional) Run on Euler or Daint on interactive session.")
  parser.set_defaults(interactive=False)
  parser.add_argument('--clockHours', type=float, default=24.0,
      help="(optional) Number of hours to allocate if running on a cluster.")
  parser.add_argument('--nTaskPerNode', type=int, default=1,
      help="(optional) Number of processes per node if running on a cluster.")

  parser.add_argument('--args', default="",
      help="(optional) Arguments to pass directly to executable")

  parsed = parser.parse_args()

  if parsed.runname is not None:
    relRunPath = parsed.runprefix + '/' + parsed.runname
  else:
    relRunPath = './'
  # rundir overwriting is allowed (exist_ok could be parsed.isTraining==False):
  os.makedirs(relRunPath, exist_ok=True)

  absRunPath = os.popen("cd "+relRunPath+" && pwd").read()[:-1] # trailing \n
  os.environ['RUNDIR'] = absRunPath

  # dir created, copy executable and read any problem-specific setup options:
  applicationSetup(parsed, absRunPath)
  # once application is defined, we can figure out all computational resouces:
  setComputationalResources(parsed)

  copySettingsFiles(parsed.settings, absRunPath)
  os.system("cd ${SMARTIES_ROOT} && git log | head  > ${RUNDIR}/gitlog.log")
  os.system("cd ${SMARTIES_ROOT} && git diff        > ${RUNDIR}/gitdiff.log")

  assert is_exe(absRunPath+'/'+parsed.execname), "FATAL: application not found"

  cmd = 'cd ${RUNDIR} \n'
  cmd = cmd + setEnvironmentFlags(parsed.nThreads);
  cmd = cmd + setLaunchCommand(parsed)
  print('COMMAND:' + parsed.args )
  os.system(cmd)


