#!/usr/bin/env python3
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
import argparse, os, psutil, sys, shutil, subprocess, signal, glob

def signal_handler(sig, frame):
  JOBID = os.getenv('SLURM_JOB_ID')
  cmd = "scancel " + JOBID
  subprocess.run(cmd, executable=parsed.shell, shell=True) 
  sys.exit(0)

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
    shutil.copytree(gymScriptPath + "pyBulletEnvironments", absRunPath + '/pyBulletEnvironments')
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
    return

  # Else user created app. First find its folder.
  if os.path.isdir( parsed.app ) or is_exe(parsed.app) :
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
    setout = subprocess.Popen(setcmd, executable=parsed.shell,
                              shell=True, stdout=subprocess.PIPE
                             ).stdout.read().splitlines()

    if len(setout[-3]):
      args = str(setout[-3], 'utf-8')
      print("app setup.sh: added args:'%s' to launch command." % args)
      parsed.args = parsed.args + " " + args

    if len(setout[-2]):
      mpiProcsPerEnv = int(setout[-2])
      print("app setup.sh: using %d MPI ranks to run each env sim." % mpiProcsPerEnv)
      if parsed.mpiProcsPerEnv > 0:
        assert ( parsed.mpiProcsPerEnv == mpiProcsPerEnv ), \
               "Contradiction between application setup and cmd line parsing"
      parsed.mpiProcsPerEnv = mpiProcsPerEnv

    if len(setout[-1]) and parsed.execname == 'exec': # if user did not specify
      execn = str(setout[-1], 'utf-8')
      print("app setup.sh: set executable name '%s'." % execn)
      parsed.execname = execn

  # elif is_exe(app):
  #   shutil.copy(app, absRunPath + '/exec')
  #   parsed.execname = 'exec'
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
    print('FATAL: Unable to locate application executable %s or %s.py'
          % (parsed.execname, parsed.execname))

  #if os.path.getmtime(app) < os.path.getmtime( SMARTIES_ROOT + '/lib/libsmarties.so'):
  #  print("WARNING: Application is older then smarties, make sure used libraries still match.")

def setComputationalResources(parsed):
  # if launched with --nProcesses N default behavior is to have N-1 workers
  if parsed.nLearners < 1 and parsed.nProcesses > 1:
    parsed.nLearners = 1
    parsed.mpiProcsPerEnv = 1

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


  if parsed.netsOnlyLearners:
    parsed.args += " --learnersOnWorkers 0 "
  if parsed.printAppStdout:
    parsed.args += " --redirectAppStdoutToFile 0 "
  if parsed.disableDataLogging:
    parsed.args += " --logAllSamples 0 "

def setTrainOrEvaluate(parsed):
  def contains_files(dirn):
      if not os.path.isdir(dirn): return False
      # whether dir contains network parameter files
      return len(glob.glob(dirn + '/agent*weights.raw')) > 0

  if parsed.nEvalEpisodes == 0:
    # training run
    parsed.args += " --nTrainSteps %d " % parsed.nTrainSteps
    if parsed.restart is None:
        parsed.args += " --restart none "
        return
  else:
    # evaluation run, we actually need to restart
    parsed.args += " --nEvalEpisodes %d " % parsed.nEvalEpisodes
    if parsed.restart is None:
      if contains_files('.'): parsed.restart = './'
      else:
        print("FATAL: Did not find restart files in current directory. "
              "Please use the option --restart")
        exit()

  if contains_files(parsed.runprefix + "/" + parsed.restart):
      dirn = parsed.runprefix + "/" + parsed.restart
      absRestartPath = os.path.abspath(dirn)
  elif contains_files(parsed.restart):
      absRestartPath = os.path.abspath(parsed.restart)
  elif contains_files(SMARTIES_ROOT + '/runs/' + parsed.restart):
      dirn = SMARTIES_ROOT + '/runs/' + parsed.restart
      absRestartPath = os.path.abspath(dirn)
  else:
    print('FATAL: Did not find restart files in %s' % parsed.restart)
    exit()

  print('Using restart files from directory %s' % absRestartPath)
  parsed.args += " --restart %s " % absRestartPath

def setEnvironmentFlags(parsed):
  env = "unset LSB_AFFINITY_HOSTFILE  #euler cluster \n" \
        "export MPICH_MAX_THREAD_SAFETY=multiple #MPICH \n" \
        "export MV2_ENABLE_AFFINITY=0 #MVAPICH \n" \
        "export OMP_NUM_THREADS=%d \n" \
        "export OPENBLAS_NUM_THREADS=1 \n" \
        "export CRAY_CUDA_MPS=1 \n" \
        "export PYTHONPATH=${PYTHONPATH}:${SMARTIES_ROOT}/lib \n" \
        "export PATH=${PATH}:${SMARTIES_ROOT}/extern/bin \n" \
        "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMARTIES_ROOT}/extern/lib \n" \
        "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib \n" \
        "export DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${SMARTIES_ROOT}/lib \n " \
        % parsed.nThreads
  if parsed.dmcApp: env += "export DISABLE_MUJOCO_RENDERING=1 \n "
  return env

def setLaunchCommand(parsed, absRunPath):
  nProcesses, nThreads, rundir = parsed.nProcesses, parsed.nThreads, parsed.runname
  clockHours = int(parsed.clockHours)
  clockMinutes = int((parsed.clockHours - clockHours) / 60)
  # default:
  #cmd = "mpirun -n %d --map-by ppr:%d:node ./%s %s | tee out.log" \
  #      % (nProcesses, parsed.nTaskPerNode, parsed.execname, parsed.args)
  cmd = "mpirun -n %d ./%s %s | tee out.log" \
        % (nProcesses, parsed.execname, parsed.args)

  if isEuler():
    assert rundir is not None, "--runname option is required on Euler and Daint"
    if   nThreads == 18 and parsed.nTaskPerNode == 1:
      map_by = "--map-by ppr:1:socket --bind-to none"
    elif nThreads == 36 and parsed.nTaskPerNode == 1:
      map_by = "--map-by ppr:1:node --bind-to none"
    else:
      map_by = "--map-by ppr:%d:node --bind-to none" % parsed.nTaskPerNode
    cmd = "mpirun -n %d %s ./%s %s " % \
          (nProcesses, map_by, parsed.execname, parsed.args)
    if parsed.interactive is False:
      cmd = "bsub -n %d -R \"select[model==XeonGold_6150] span[ptile=36]\" " \
          " -J %s -W %s:00 %s " \
          % (nProcesses * nThreads, rundir, clockHours, cmd )

  elif isDaint() and parsed.interactive is False:
    nTaskPerNode, nNodes = parsed.nTaskPerNode, nProcesses / parsed.nTaskPerNode
    assert rundir is not None, "--runname option is required on Euler and Daint"
    f = open(absRunPath + '/daint_sbatch','w')
    f.write('#!/bin/bash -l \n')
    f.write('#SBATCH --job-name=%s \n' % rundir)
    if parsed.debug:
      f.write('#SBATCH --time=00:30:00 \n')
      f.write('#SBATCH --partition=debug \n')
    else:
      f.write('#SBATCH --time=%s:00:00 \n' % clockHours)
    f.write('#SBATCH --output=%s_out_%%j.txt \n' % rundir)
    f.write('#SBATCH --error=%s_err_%%j.txt \n'  % rundir)
    f.write('#SBATCH --constraint=gpu \n')
    f.write('#SBATCH --account=s929 \n')
    f.write('#SBATCH --nodes=%d \n' % nNodes)
    f.write('srun -n %d --nodes=%d --ntasks-per-node=%d ./%s %s \n' \
            % (nProcesses, nNodes, nTaskPerNode, parsed.execname, parsed.args))
    f.close()
    cmd = "chmod 755 daint_sbatch \n sbatch daint_sbatch"

  elif isDaint() and parsed.interactive is True:
    nTaskPerNode, nNodes = parsed.nTaskPerNode, nProcesses / parsed.nTaskPerNode
    cmd = "srun -C gpu -u -p debug -n %d --nodes %d --ntasks-per-node %d ./%s %s" \
          % (nProcesses, nNodes, nTaskPerNode, parsed.execname, parsed.args)
  return cmd

if __name__ == '__main__':
  assert len(SMARTIES_ROOT)>0, \
         "FATAL: Environment variable SMARTIES_ROOT is unset. Read the README"
  assert os.path.isfile(SMARTIES_ROOT+'/lib/libsmarties.so') or \
         os.path.isfile(SMARTIES_ROOT+'/lib/libsmarties.dylib'), \
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
      help="path or name of the settings file specifying RL solver " \
           "and its hyper-parameters. The default setting file is set to VRACER.json")

  parser.add_argument('-r','--runname', default=None,
      help="Name of the directory in which the learning process will be executed. " \
           "If unset, execution will take place in the current directory.")

  parser.add_argument('--nThreads', type=int, default=nThreads,
      help="Number of threads used by the learning processes. " \
           "The default value is the number of available CPU cores, here %d." \
           % nThreads)
  parser.add_argument('-n','--nProcesses',     type=int, default=0, # 0 tells me no expressed preference
      help="Number of processes available to run the training.")
  parser.add_argument('-l','--nLearners',      type=int, default=0, # 0 tells me no expressed preference
      help="Number of processes dedicated to update the networks. By default 1.")
  parser.add_argument('-e','--nEnvironments',  type=int, default=0, # 0 tells me no expressed preference
      help="Number of concurrent environment simulations. By default 1.")
  parser.add_argument('-m','--mpiProcsPerEnv', type=int, default=0, # 0 tells me no expressed preference
    help="MPI processes required per env simulation. This value can also " \
         "be specified in app's setup.sh script by setting the MPI_RANKS_PER_ENV " \
         "shell variable. If unset or 0, smarties performs communication via " \
         "sockets and avoids creating multiple MPI processes.")

  parser.add_argument('--netsOnlyOnLearners', dest='netsOnlyLearners', action='store_true',
    help="Forces network approximator to live only inside learning " \
         "processes. If this option is used, workers send states to learners " \
         "and learners reply with actions. Otherwise, workers collect entire " \
         "episodes, send them to learners and receive parameter updates.")
  parser.set_defaults(netsOnlyLearners=False)

  parser.add_argument('--printAppStdout', dest='printAppStdout', action='store_true',
    help="Prints application output to screen. If unset, application " \
         "output will be redirected to file in the simulation subfolder.")
  parser.set_defaults(printAppStdout=False)

  parser.add_argument('--disableDataLogging', dest='disableDataLogging', action='store_true',
    help="Stops smarties from storing all state/action/reward/policy " \
         "into (binary) log files. These files enable postprocessing and analysis " \
         " but may occupy a lot of storage space.")
  parser.set_defaults(disableDataLogging=False)

  parser.add_argument('--restart', default=None,
      help="Path to existing directory which contains smarties output files "
           "needed to restart already trained agents.")
  parser.add_argument('-t','--nTrainSteps', type=int, default=100000000,
      help="Total number of time steps before end of learning.")
  parser.add_argument('--nEvalEpisodes', type=int, default=0,
      help="Number of environment episodes to evaluate trained policy. " \
           "This option automatically disables training.")

  parser.add_argument('--gym',   dest='gymApp',   action='store_true',
    help="Set if application is part of OpenAI gym.")
  parser.set_defaults(gymApp=False)
  parser.add_argument('--atari', dest='atariApp', action='store_true',
    help="Set if application is part of OpenAI gym's atari suite.")
  parser.set_defaults(atariApp=False)
  parser.add_argument('--dmc',   dest='dmcApp',   action='store_true',
    help="Set if application is part of DeepMind control suite.")
  parser.set_defaults(dmcApp=False)

  parser.add_argument('--execname',  default='exec',
      help="Name of application's executable.")
  parser.add_argument('--runprefix', default=runprefix,
      help="Path to directory where run folder will be created.")

  parser.add_argument('--interactive', dest='interactive', action='store_true',
    help="Run on Euler or Daint on interactive session.")
  parser.set_defaults(interactive=False)
  parser.add_argument('--debug', dest='debug', action='store_true',
    help="Run on Daint on debug partition.")
  parser.set_defaults(debug=False)
  parser.add_argument('--clockHours', type=float, default=24.0,
      help="Number of hours to allocate if running on a cluster.")
  parser.add_argument('--nTaskPerNode', type=int, default=1,
      help="Number of processes per node if running on a cluster.")
  parser.add_argument('--shell', type=str, default='/bin/bash',
      help="Which shell will be used to execute launch command. " \
           "Defaults to /bin/bash.")

  parser.add_argument('--args', default="",
      help="Arguments to pass directly to executable")

  parsed = parser.parse_args()

  if parsed.runname is not None:
    relRunPath = parsed.runprefix + '/' + parsed.runname
  else:
    relRunPath = './'
  # rundir overwriting is allowed (exist_ok could be parsed.isTraining==False):
  os.makedirs(relRunPath, exist_ok=True)

  absRunPath = os.path.abspath(relRunPath)
  os.environ['RUNDIR'] = absRunPath

  copySettingsFiles(parsed.settings, absRunPath)
  # dir created, copy executable and read any problem-specific setup options:
  applicationSetup(parsed, absRunPath)
  # once application is defined, we can figure out all computational resouces:
  setComputationalResources(parsed)
  # define how many training steps, evaluation episodes, where/if to find restart
  setTrainOrEvaluate(parsed)

  subprocess.run("cd ${SMARTIES_ROOT} && git log | head > ${RUNDIR}/gitlog.log", \
                 executable=parsed.shell, shell=True)
  subprocess.run("cd ${SMARTIES_ROOT} && git diff       > ${RUNDIR}/gitdiff.log", \
                 executable=parsed.shell, shell=True)

  assert is_exe(absRunPath+'/'+parsed.execname), "FATAL: application not found"

  cmd = 'cd ${RUNDIR} \n'
  cmd = cmd + setEnvironmentFlags(parsed)
  cmd = cmd + setLaunchCommand(parsed, absRunPath)

  # print('COMMAND:' + cmd )
  signal.signal(signal.SIGINT, signal_handler)
  subprocess.run(cmd, executable=parsed.shell, shell=True)
