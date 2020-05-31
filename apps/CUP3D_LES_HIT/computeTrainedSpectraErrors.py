#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from computeSpectraNonDim     import readAllSpectra
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
#colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#006837', '#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43', '#d73027', '#a50026', '#8e0152', '#c51b7d', '#de77ae', '#f1b6da']

def findDirectory(runspath, re, token):
    retoken = 'RE%03d' % re
    alldirs = glob.glob(runspath + '/*')
    for dirn in alldirs:
        if retoken not in dirn: continue
        if token not in dirn: continue
        return dirn
    assert(False, 're-token combo not found')

def main_integral(runspath, target, REs, tokens, labels):
    nBins = 2 * 16//2 - 1
    modes = np.arange(1, nBins+1, dtype=np.float64) # assumes box is 2 pi
    plt.figure()
    #REs = findAllParams(path)
    nRes = len(REs)
    axes, lines = [], []
    for j in range(nRes):
        axes += [ plt.subplot(1, nRes, j+1) ]

    for j in range(nRes):
        RE = REs[j]
        # read target file
        logSpectra, logEnStdev, _, _ = readAllSpectra(target, [RE])
        for i in range(len(tokens)):
            eps, nu = epsNuFromRe(RE)
            dirn = findDirectory(runspath, RE, tokens[i])
            runData = getAllData(dirn, eps, nu, nBins, fSkip=1)
            logE = np.log(runData['spectra'])
            avgLogSpec = np.mean(logE, axis=0)
            assert(avgLogSpec.size == nBins)
            LL = (avgLogSpec.ravel() - logSpectra.ravel()) / logEnStdev.ravel()
            print(LL.shape)
            p = axes[j].plot(LL, modes, label=labels[i], color=colors[i])
            #p = axes[j].plot(LL, modes, color=colors[i])
            if j == 0: lines += [p]
            #stdLogSpec = np.std(logE, axis=0)
            #covLogSpec = np.cov(logE, rowvar=False)
            #print(covLogSpec.shape)
    axes[0].set_ylabel(r'$k$')
    for j in range(nRes):
        axes[j].set_title(r'$Re_\lambda$ = %d' % REs[j])
        #axes[j].set_xscale("log")
        axes[j].set_ylim([1, 15])
        axes[j].grid()
        axes[j].set_xlabel(r'$\frac{\log E(k) - \mu_{\log E(k)}}{\sigma_{\log E(k)}}$')
    for j in range(1,nRes): axes[j].set_yticklabels([])
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    assert(len(lines) == len(labels))
    #axes[0].legend(lines, labels, bbox_to_anchor=(0.5, 0.5))
    axes[0].legend(bbox_to_anchor=(0.5, 0.5))
    plt.tight_layout()
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
  assert(len(args.tokens) == len(args.labels))

  main_integral(args.runspath, args.target, args.res, args.tokens, args.labels)
