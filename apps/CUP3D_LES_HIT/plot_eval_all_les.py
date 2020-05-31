#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, subprocess, time
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from computeSpectraNonDim     import readAllSpectra

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#colors = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
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
  bestDir, bestLL, bestStats = None, -1e9, None
  filen1 = '/simulation_000_00000/run_00000000/spectrumProbability.text'
  filen2 = '/spectrumProbability.text'
  if token[0] == '/': alldirs = glob.glob(token + '*' + ('RE%03d' % re) + '*')
  else:               alldirs = glob.glob(path + '/*' + ('RE%03d' % re) + '*')
  for dirn in alldirs:
    if token not in dirn: continue
    #print (dirn)
    # if token did not produce any stats file (e.g. not applicable to dns):
    if bestDir is None: bestDir = dirn
    # else, pick best perfoming alternative:
    if   os.path.isfile(dirn+filen1): stats = np.fromfile(dirn+filen1, sep=' ')
    elif os.path.isfile(dirn+filen2): stats = np.fromfile(dirn+filen2, sep=' ')
    else : continue
    if len(stats) == 0: continue
    #print(dirn, stats)
    if stats[1] > bestLL and stats[3] > 1000:
        bestDir, bestLL, bestStats = dirn, stats[1], stats
  assert bestDir is not None, "token %s - %d not found" % (token, re)
  #print(bestDir, bestStats)
  return bestDir

def findDirectory(runspath, re, token):
    retoken = 'RE%03d' % re
    alldirs = glob.glob(runspath + '/*')
    for dirn in alldirs:
        if retoken not in dirn: continue
        if token not in dirn: continue
        return dirn
    assert False, 're-token combo not found'

def main_integral(runspath, target, REs, tokens, labels, nBins):
    minNbins = min(nBins)
    #plt.figure()
    #REs = findAllParams(path)
    nRes = len(REs)
    axes, lines = [], []
    fig, axes = plt.subplots(2, nRes, sharey=True, figsize=[12, 4.8], frameon=False, squeeze=True)
    #for j in range(nRes):
    #    axes += [ plt.subplot(1, nRes, j+1) ]

    for j in range(nRes):
        RE = REs[j]
        eps, nu = epsNuFromRe(RE)
        Ekscal = np.power(nu**5 * eps, 0.25)

        # read target file
        logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
        logSpectra = logSpectra.reshape([logSpectra.size])
        logEnStdev = logEnStdev.reshape([logSpectra.size])
        minNbins = min(minNbins, logSpectra.size)
        modes = np.arange(1, minNbins+1, dtype=np.float64) # assumes box is 2 pi
        axes[1][j].plot(np.zeros(minNbins), modes, 'k-')
        LLTop = np.exp(logSpectra + logEnStdev)/Ekscal
        LLBot = np.exp(logSpectra - logEnStdev)/Ekscal
        axes[0][j].plot(np.exp(logSpectra)/Ekscal, modes, color='k')
        axes[0][j].fill_betweenx(modes, LLBot, LLTop, facecolor='k', alpha=.5)

        for i in range(len(tokens)):
            dirn = findBestHyperParams(runspath, RE, tokens[i])
            print(nBins[i])
            runData = getAllData(dirn, eps, nu, nBins[i], fSkip=1)
            logE = np.log(runData['spectra'])
            print(logE.shape[0])
            avgLogSpec, stdLogSpec = np.mean(logE, axis=0), np.std(logE, axis=0)
            assert(avgLogSpec.size == nBins[i])
            avgLogSpec = avgLogSpec[:minNbins]
            stdLogSpec = stdLogSpec[:minNbins]
            #print(avgLogSpec.shape, logSpectra.shape, logEnStdev.shape)
            LL = (avgLogSpec - logSpectra) / logEnStdev
            p = axes[1][j].plot(LL, modes, label=labels[i], color=colors[i]) # , label=labels[i]
            LLTop = (avgLogSpec + stdLogSpec - logSpectra) / logEnStdev
            LLBot = (avgLogSpec - stdLogSpec - logSpectra) / logEnStdev
            axes[1][j].fill_betweenx(modes, LLBot, LLTop, facecolor=colors[i], alpha=.5)

            Ek = np.exp(avgLogSpec) / Ekscal
            axes[0][j].plot(Ek, modes, label=labels[i], color=colors[i]) # , label=labels[i]
            LLTop = np.exp(avgLogSpec+stdLogSpec) / Ekscal
            LLBot = np.exp(avgLogSpec-stdLogSpec) / Ekscal
            axes[0][j].fill_betweenx(modes, LLBot, LLTop, facecolor=colors[i], alpha=.5)
            '''
              LLt = (0.5 * (logE - logSpectra) / logEnStdev ) ** 2
              sumLLt = np.sum(LLt, axis=1)
              nSamples = sumLLt.size
              print('found %d samples' % nSamples)
              p = axes[j].plot(np.arange(nSamples), sumLLt, label=labels[i], color=colors[i])
            '''
            if j == 0: lines += [p]
    axes[0][0].set_ylabel(r'$k \cdot L / 2 \pi$')
    axes[1][0].set_ylabel(r'$k \cdot L / 2 \pi$')

    for j in range(nRes):
        axes[0][j].set_title(r'$Re_\lambda$ = %d' % REs[j])
        axes[0][j].set_xscale("log")
        axes[0][j].set_ylim([1, 15])
        axes[1][j].set_ylim([1, 15])
        axes[0][j].grid()
        axes[1][j].grid()
        #axes[1][j].set_xlabel(r'$\frac{\log E^{LES}(k) - \mu[\log E^{DNS}(k)]}{\sigma[\log E^{DNS}(k)]}$')
        axes[1][j].set_xlabel(r'$\frac{\log E_{LES} - \mu(\log E_{DNS})}{\sigma(\log E_{DNS})}$')
        axes[0][j].set_xlabel(r'$E_{LES} \,/\, \eta u^2_\eta$')
    
    axes[0][0].invert_yaxis()
    #axes[1][0].invert_yaxis()
    #for j in range(1,nRes): axes[j].set_yticklabels([])
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    assert(len(lines) == len(labels))
    #axes[0].legend(lines, labels, bbox_to_anchor=(0.5, 0.5))
    #fig.subplots_adjust(right=0.17, wspace=0.2)
    #axes[0][-1].legend(bbox_to_anchor=(0.25, 0.25), borderaxespad=0)
    #axes[-1].legend(bbox_to_anchor=(1, 0.5),fancybox=False, shadow=False)
    #fig.legend(lines, labels, loc=7, borderaxespad=0)
    fig.tight_layout()
    #fig.subplots_adjust(right=0.75)
    plt.show()
    #axes[0].legend(loc='lower left')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  ADD = parser.add_argument
  ADD('tokens', nargs='+',
    help="Directory naming tokens shared by all runs across Re")
  ADD('--target', default='target_RKBPD32_2blocks/',
    help="Path to target DNS files directory")
  ADD('--res', nargs='+', type=int, default = [65, 76, 88, 103, 120, 140, 163],
    help="Reynolds numbers to visualize")
  ADD('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  ADD('--runspath', default='../../runs/',
    help="Relative path to evaluation runs")
  ADD('--gridSize', nargs='+', type=int, default=[32],
    help="1D grid size used by the evaluation runs")

  args = parser.parse_args()
  if args.labels is None: args.labels = args.tokens
  if len(args.gridSize) < len(args.labels):
      assert(len(args.gridSize) == 1)
      args.gridSize = args.gridSize * len(args.labels)
  assert len(args.labels) == len(args.tokens)
  for i in range(len(args.gridSize)): args.gridSize[i] = args.gridSize[i]//2 - 1

  main_integral(args.runspath, args.target, args.res, args.tokens, args.labels, args.gridSize)



