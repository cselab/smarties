#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, subprocess, time
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from computeSpectraNonDim     import readAllSpectra

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
#colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026', '#8e0152', '#c51b7d', '#de77ae', '#f1b6da']

lastCompiledBlockSize = 16

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

'''
def findBestHyperParams(path, token, re):
  bestDir, bestLL = -1, -1e9
  for nBlocks in [2, 4, 8]:
    for actFreq in [2, 4, 8, 16]:
      for nn in ['FFNN']:
        re_spec = 'RE%03d' % re
        hp_spec = '%s_%dblocks_act%02d' % (nn, nBlocks, actFreq)
        alldirs = glob.glob(path + '/*' + token + '*' + hp_spec + '*' + re_spec)
        for dirn in alldirs:
          print(dirn)
          filen = '/simulation_000_00000/run_00000000/spectrumProbability.text'
          stats = np.fromfile(dirn + finen, sep=' ')
          if stats[1] > bestLL: bestDir, bestLL = dirn, stats[1]
  return bestDir
'''

def findBestHyperParams(path, re, token):
  bestDir, bestLL = None, -1e9
  filen1 = '/simulation_000_00000/run_00000000/spectrumProbability.text'
  filen2 = '/spectrumProbability.text'
  alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
  for dirn in alldirs:
    if token not in dirn: continue
    if   os.path.isfile(dirn+filen1): stats = np.fromfile(dirn+filen1, sep=' ')
    elif os.path.isfile(dirn+filen2): stats = np.fromfile(dirn+filen2, sep=' ')
    else : continue
    if len(stats) == 0: continue
    if stats[1] > bestLL: bestDir, bestLL = dirn, stats[1]
  assert bestDir is not None, "token %s not found" % token
  print(bestDir)
  return bestDir

def findDirectory(runspath, re, token):
    retoken = 'RE%03d' % re
    alldirs = glob.glob(runspath + '/*')
    for dirn in alldirs:
        if retoken not in dirn: continue
        if token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main_integral(runspath, target, REs, tokens, labels):
    nBins = 2 * 16//2 - 1
    modes = np.arange(1, nBins+1, dtype=np.float64) # assumes box is 2 pi
    #plt.figure()
    #REs = findAllParams(path)
    nRes = len(REs)
    axes, lines = [], []
    fig, axes = plt.subplots(1, nRes, sharex=True, sharey=True, figsize=[12, 4.8])
    #for j in range(nRes):
    #    axes += [ plt.subplot(1, nRes, j+1) ]

    for j in range(nRes):
        RE = REs[j]
        # read target file
        logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
        logSpectra = logSpectra.reshape([logSpectra.size])
        logEnStdev = logEnStdev.reshape([logSpectra.size])
        axes[j].plot(np.zeros(nBins), modes, 'k-')
        for i in range(len(tokens)):
            eps, nu = epsNuFromRe(RE)
            dirn = findBestHyperParams(runspath, RE, tokens[i])
            runData = getAllData(dirn, eps, nu, nBins, fSkip=1)
            logE = np.log(runData['spectra'])
            print(logE.shape[0])
            avgLogSpec, stdLogSpec = np.mean(logE, axis=0), np.std(logE, axis=0)
            assert(avgLogSpec.size == nBins)
            LL = (avgLogSpec - logSpectra) / logEnStdev
            p = axes[j].plot(LL, modes, label=labels[i], color=colors[i]) # , label=labels[i]
            LLTop = (avgLogSpec + stdLogSpec - logSpectra) / logEnStdev
            LLBot = (avgLogSpec - stdLogSpec - logSpectra) / logEnStdev
            axes[j].fill_betweenx(modes, LLBot, LLTop, facecolor=colors[i], alpha=.5)
            '''
              LLt = (0.5 * (logE - logSpectra) / logEnStdev ) ** 2
              sumLLt = np.sum(LLt, axis=1)
              nSamples = sumLLt.size
              print('found %d samples' % nSamples)
              p = axes[j].plot(np.arange(nSamples), sumLLt, label=labels[i], color=colors[i])
            '''
            if j == 0: lines += [p]
    axes[0].set_ylabel(r'$k$')
    for j in range(nRes):
        axes[j].set_title(r'$Re_\lambda$ = %d' % REs[j])
        #axes[j].set_xscale("log")
        axes[j].set_ylim([1, 15])
        axes[j].grid()
        axes[j].invert_yaxis()
        axes[j].set_xlabel(r'$\frac{\log E(k) - \mu^{DNS}_{\log E(k)}}{\sigma^{DNS}_{\log E(k)}}$')
    #for j in range(1,nRes): axes[j].set_yticklabels([])
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    assert(len(lines) == len(labels))
    #axes[0].legend(lines, labels, bbox_to_anchor=(0.5, 0.5))
    #fig.subplots_adjust(right=0.17, wspace=0.2)
    #axes[-1].legend(bbox_to_anchor=(0.25, 0.25), borderaxespad=0)
    #axes[-1].legend(bbox_to_anchor=(1, 0.5),fancybox=False, shadow=False)
    #fig.legend(lines, labels, loc=7, borderaxespad=0)
    fig.tight_layout()
    #fig.subplots_adjust(right=0.75)
    plt.show()
    #axes[0].legend(loc='lower left')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--target', help="Path to target files directory")
  parser.add_argument('--tokens', nargs='+', help="Text token distinguishing each series of runs")
  parser.add_argument('--res', nargs='+', type=int, help="Reynolds numbers")
  parser.add_argument('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  parser.add_argument('--runspath', help="Plot labels to assiciate to tokens")

  args = parser.parse_args()
  if args.labels is None: args.labels = args.tokens
  assert len(args.labels) == len(args.tokens)

  main_integral(args.runspath, args.target, args.res, args.tokens, args.labels)

def launchEvalRun(traindir, targetsPath):
  envVars  = 'export LES_RL_NETTYPE=FFNN\n export SKIPMAKE=true\n '
  envVars += 'export LES_RL_N_TSIM=1000\n '
  envVars += 'export LES_RL_GRIDACT=1\n '
  if findIfGridAgent(traindir):
        envVars += 'export LES_RL_GRIDACTSETTINGS=1\n '
  else: envVars += 'export LES_RL_GRIDACTSETTINGS=0\n '
  envVars += 'export LES_RL_NBLOCK=%d\n ' % findBlockNum(traindir)
  envVars += 'export LES_RL_FREQ_A=%d\n ' % findActFreq(traindir)

  for re in [60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205]:
    envVars += 'export LES_RL_EVALUATE=RE%03d\n ' % (re)
    dirname = '%s_RE%03d' % (traindir, re)
    options = 'CUP3D_LES_HIT -n 1 --nEvalSeqs 1 --clockHours 1 --nTaskPerNode 2'
    trainPath = targetsPath + '/' + traindir
    cmd = 'smarties.py %s -r %s --restart %s ' % (options, dirname, trainPath)
    if os.path.isdir( dirname ):
      print('%s already exists, skipping' % dirname)
    else:
      subprocess.run(envVars + cmd, shell=True)
      time.sleep(10) # prevent daint from disabling job submission

'''
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targetsPath',
    help="Simulation directories containing the trained policies")
  parser.add_argument('targets', nargs='+',
    help="Simulation directories containing the trained policies")
  args = parser.parse_args()
  for dirn in args.targets: launchEvalRun(dirn, args.targetsPath)
'''
#THISDIR=${SMARTIES_ROOT}/apps/CUP3D_LES_HIT
#SPEC=${REW}_${LES_RL_NETTYPE}_${LES_RL_NBLOCK}blocks_act`printf %02d $LES_RL_FREQ_A`
