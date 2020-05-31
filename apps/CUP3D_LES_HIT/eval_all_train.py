#!/usr/bin/env python3
import os, numpy as np, argparse, subprocess

def findIfGridAgent(traindir):
  if 'BlockAgents' in traindir: return False
  return True
def findActFreq(traindir):
  if 'act02' in traindir: return 2
  if 'act04' in traindir: return 4
  if 'act08' in traindir: return 8
  if 'act16' in traindir: return 16
  assert False
  return 0
def findBlockSize(traindir):
  if '2blocks' in traindir: return 16
  if '4blocks' in traindir: return 8
  if '8blocks' in traindir: return 4
  assert False
  return 0
def findBlockNum(traindir):
  if '2blocks' in traindir: return 2
  if '4blocks' in traindir: return 4
  if '8blocks' in traindir: return 8
  assert False
  return 0

def launch(dirn, path, useBlockNumber, bGridAgents):
  cmd = ''
  #cmd = cmd + ' export SKIPMAKE=true \n '
  cmd = cmd + ' export LES_RL_N_TSIM=100 \n '
  cmd = cmd + (' export LES_RL_FREQ_A=%d \n ' % findActFreq(dirn))
  cmd = cmd + (' export LES_RL_NBLOCK=%d \n ' % useBlockNumber)
  if bGridAgents: cmd = cmd + ' export LES_RL_GRIDACT=1 \n '
  else:           cmd = cmd + ' export LES_RL_GRIDACT=0 \n '
  cmd = cmd + ' export LES_RL_NETTYPE=FFNN \n '
  cmd = cmd + ' export LES_RL_GRIDACTSETTINGS=0 \n '
  common = ' smarties.py CUP3D_LES_HIT --nEvalEpisodes 2 --clockHours 1 --nTaskPerNode 2 -n 1'
  for re in [60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205]:
    cmdre = cmd + ' export LES_RL_EVALUATE=RE%03d \n ' % re
    #runn = '%s_%03dPD_RE%03d' % (dirn, 16 * useBlockNumber, re)
    runn = '%s_RE%03d' % (dirn, re)
    runcmd = '%s %s -r %s --restart %s' % (cmdre, common, runn, path+'/'+dirn)
    #print(runcmd)
    subprocess.run(runcmd, shell=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Evaluate trained directories.")
  parser.add_argument('restarts', nargs='+',
                      help="Directories containing trained policy to evaluate")
  parser.add_argument('--restartsPath', default='../../runs/',
                      help="Optional path to trained dirs, if not default")
  parser.add_argument('--useBlockNumber', type=int, default=4,
                      help="Number of cubismup3d blocks to use.")
  parser.add_argument('--bGridAgents', dest='bGridAgents', action='store_true',
    help="Force one agent per grid point.")
  parser.set_defaults(bGridAgents=False)
  args = parser.parse_args()

  for dirn in args.restarts:
    launch(dirn, args.restartsPath, args.useBlockNumber, args.bGridAgents)