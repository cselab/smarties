#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, subprocess, time
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from computeSpectraNonDim     import readAllSpectra
from plot_eval_all_les import findBestHyperParams as findBestHyperParamsReference

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
#colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026', '#8e0152', '#c51b7d', '#de77ae', '#f1b6da']
colors = ['#6baed6','#3182bd','#08519c', '#74c476','#31a354','#006d2c']
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

def findBestHyperParams(path, re, token, gridSize):
  bestDir, bestLL, bestStats = None, -1e9, None
  filen1 = '/simulation_000_00000/run_00000000/spectrumProbability.text'
  filen2 = '/spectrumProbability.text'
  alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
  gridToken  = '%03dPD' % gridSize
  for dirn in alldirs:
    if token not in dirn: continue
    #if gridSize == 32 and '128PD' in dirn: continue 
    #if gridSize == 32 and '064PD' in dirn: continue
    #elif gridSize != 32 and gridToken not in dirn: continue 
    if gridToken not in dirn: continue
    #print (dirn)
    if   os.path.isfile(dirn+filen1): stats = np.fromfile(dirn+filen1, sep=' ')
    elif os.path.isfile(dirn+filen2): stats = np.fromfile(dirn+filen2, sep=' ')
    else : continue
    if len(stats) == 0: continue
    #print(dirn, stats)
    if stats[1] > bestLL and stats[3] > 1000:
        bestDir, bestLL, bestStats = dirn, stats[1], stats
  assert bestDir is not None, "token %s - %d not found" % (token, re)
  print(bestDir, bestStats)
  return bestDir

def findDirectory(runspath, re, token):
    retoken = 'RE%03d' % re
    alldirs = glob.glob(runspath + '/*')
    for dirn in alldirs:
        if retoken not in dirn: continue
        if token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main_integral(runspath, target, REs, tokens, labels, gridSizes, refs):
    nBins = np.zeros(len(gridSizes), dtype=int)
    for i in range(len(gridSizes)): nBins[i] = gridSizes[i]//2 - 1
    maxNbins, nRes = max(nBins), len(REs)

    axes, lines = [], []
    fig, axes = plt.subplots(1, nRes, sharey=True, figsize=[12, 3], frameon=False, squeeze=True)

    for j in range(nRes):
        RE, colorid = REs[j], 0
        eps, nu = epsNuFromRe(RE)
        logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
        logSpectra = logSpectra.reshape([logSpectra.size])
        logEnStdev = logEnStdev.reshape([logSpectra.size])

        modes = np.arange(1, maxNbins+1, dtype=np.float64) # assumes box is 2 pi
        axes[j].plot(np.zeros(maxNbins), modes, 'k-')

        for i in range(len(tokens)):
          for k in range(len(gridSizes)):
            dirn = findBestHyperParams(runspath, RE, tokens[i], gridSizes[k])
            runData = getAllData(dirn, eps, nu, nBins[k], fSkip=1)
            logE = np.log(runData['spectra'])

            avgLogSpec, stdLogSpec = np.mean(logE, axis=0), np.std(logE, axis=0)
            assert(avgLogSpec.size == nBins[k])

            modes = np.arange(1, nBins[k]+1, dtype=np.float64) # assumes box is 2 pi
            M, S = logSpectra[:nBins[k]], logEnStdev[:nBins[k]]
            LL = (avgLogSpec - M) / S
            color, colorid = colors[colorid], colorid + 1
            p = axes[j].plot(LL, modes, label=labels[i], color=color) # , label=labels[i]
            LLTop = (avgLogSpec + stdLogSpec - M) / S
            LLBot = (avgLogSpec - stdLogSpec - M) / S
            axes[j].fill_betweenx(modes, LLBot, LLTop, facecolor=color, alpha=.5)

            if j * k == 0: lines += [p]

        for i in range(len(refs)):
            dirn = findBestHyperParamsReference(runspath, RE, refs[i])
            if 'BPD2' in dirn: gridsize =  32
            if 'BPD4' in dirn: gridsize =  64
            if 'BPD8' in dirn: gridsize = 128
            nBin = gridsize//2 - 1
            runData = getAllData(dirn, eps, nu, nBin, fSkip=1)
            logE = np.log(runData['spectra'])

            avgLogSpec, stdLogSpec = np.mean(logE, axis=0), np.std(logE, axis=0)
            assert(avgLogSpec.size == nBin)

            modes = np.arange(1, nBin+1, dtype=np.float64) # assumes box is 2 pi
            M, S = logSpectra[:nBin], logEnStdev[:nBin]
            LL = (avgLogSpec - M) / S
            color, colorid = colors[colorid], colorid + 1
            p = axes[j].plot(LL, modes, color=color) # , label=labels[i]
            LLTop = (avgLogSpec + stdLogSpec - M) / S
            LLBot = (avgLogSpec - stdLogSpec - M) / S
            axes[j].fill_betweenx(modes, LLBot, LLTop, facecolor=color, alpha=.5)

    axes[0].set_ylabel(r'$k \cdot L / 2 \pi$')
    axes[0].set_yscale("symlog", linthreshy=15)
    axes[0].set_ylim([1, maxNbins])
    axes[0].set_yticks([10, 15, 31, 63])
    axes[0].set_yticklabels(['10', '15', '31', '63'])
    axes[0].invert_yaxis()

    for j in range(nRes):
      axes[j].grid()
      axes[j].set_title(r'$Re_\lambda$ = %d' % REs[j])
      axes[j].set_xlabel(r'$\frac{\log E_{LES} - \mu(\log E_{DNS})}{\sigma(\log E_{DNS})}$')

    axes[0].set_xlim([-6.2, 2.9])
    axes[1].set_xlim([-6.7, 5.0])
    axes[2].set_xlim([-7.2, 7])
    axes[3].set_xlim([-6.5, 8.5])
    axes[4].set_xlim([-7.2, 10])
    axes[5].set_xlim([-6.8, 11.5])
    axes[6].set_xlim([-6.9, 13])
    #assert(len(lines) == len(labels))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  ADD = parser.add_argument
  ADD('--target', help="Path to target files directory",
    default='/users/novatig/smarties/apps/CUP3D_LES_HIT/target_RK_8blocks/')
  ADD('--tokens', nargs='+', help="Text token distinguishing each series of runs")
  ADD('--refs', nargs='+', help="Text token distinguishing each series of runs")
  ADD('--res', nargs='+', type=int, help="Reynolds numbers",
    default = [60, 70, 111, 151, 176, 190, 205])
  ADD('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  ADD('--runspath', default='./', help="Plot labels to assiciate to tokens")
  ADD('--gridSize', nargs='+', type=int, default=[32], help="Plot labels to assiciate to tokens")

  args = parser.parse_args()

  if args.labels is None: args.labels = args.tokens
  assert len(args.labels) == len(args.tokens)

  #if len(args.gridSize) < len(args.labels):
  #    assert(len(args.gridSize) == 1)
  #    args.gridSize = args.gridSize * len(args.labels)

  main_integral(args.runspath, args.target, args.res, args.tokens, args.labels, args.gridSize, args.refs)



