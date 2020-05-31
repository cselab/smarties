#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, numpy as np, os
from plot_eval_all_les import findBestHyperParams
from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData
from computeSpectraNonDim     import readAllSpectra

colors = ['#377eb8', '#e41a1c', '#ff7f00', '#4daf4a', '#984ea3', '#ffff33', '#a65628', '#f781bf', '#999999']
nBlocks = [2, 4,  8]
nAgents = [8, 64, 512]
nAgentsLab = ['8', '64', '512']
actFreqLab = ['4', '8', '16']
actFreq = [4, 8, 16]

def readAvgStdLL(dirn, re, logSpectra, logEnStdev):
  '''
  filen1 = '/simulation_000_00000/run_00000000/spectrumProbability.text'
  filen2 = '/spectrumProbability.text'
  if   os.path.isfile(dirn+filen1): stats = np.fromfile(dirn+filen1, sep=' ')
  elif os.path.isfile(dirn+filen2): stats = np.fromfile(dirn+filen2, sep=' ')
  else : assert False, 'not found %s' % dirname
  return stats[1], stats[2]
  '''
  eps, nu = epsNuFromRe(re)
  runData = getAllData(dirn, eps, nu, 2 * 16//2 - 1, fSkip=1)
  logE = np.log(runData['spectra'])
  LLt = - np.sum(0.5 * ((logE - logSpectra)/logEnStdev) ** 2, axis=1)
  print('%s found %d samples' % (dirn, LLt.size) )
  return np.mean(LLt, axis=0), np.std(LLt, axis=0)
  #'''

def main(runspath, REs, tokens, labels, target):
    nRes = len(REs)
    fig, axes = plt.subplots(2,nRes, sharey='row', figsize=[12, 4], frameon=False, squeeze=True)

    for j in range(nRes):
      logSpectra, logEnStdev, _, _ = readAllSpectra(target, [REs[j]])
      logSpectra = logSpectra.reshape([logSpectra.size])
      logEnStdev = logEnStdev.reshape([logSpectra.size])
      for i in range(len(tokens)):
        afLLmean, afLLstd = np.zeros(len(actFreq)), np.zeros(len(actFreq))
        nbLLmean, nbLLstd = np.zeros(len(nBlocks)), np.zeros(len(nBlocks))
        for k in range(len(actFreq)):
          RE, af = REs[j], actFreq[k]
          tok = '%s_%dblocks_act%02d' % (tokens[i], 4, af)
          dirn = findBestHyperParams(runspath, RE, tok)
          M, S = readAvgStdLL(dirn, RE, logSpectra, logEnStdev)
          afLLmean[k], afLLstd[k] = M, S

        for k in range(len(nBlocks)):
          RE, nb = REs[j], nBlocks[k]
          tok = '%s_%dblocks_act%02d' % (tokens[i], nb, 8)
          dirn = findBestHyperParams(runspath, RE, tok)
          M, S = readAvgStdLL(dirn, RE, logSpectra, logEnStdev)
          nbLLmean[k], nbLLstd[k] = M, S

        axes[0][j].errorbar(actFreq, afLLmean, fmt='o', yerr=afLLstd,
          color=colors[i], capsize=5, elinewidth=2, markeredgewidth=2)
        axes[1][j].errorbar(nAgents, nbLLmean, fmt='o', yerr=nbLLstd,
          color=colors[i], capsize=5, elinewidth=2, markeredgewidth=2)
        #Yb, Yt = afLLmean[k] - afLLstd[k], afLLmean[k] + afLLstd[k]
        #axes[0][j].fill_between(actFreq, Yb, Yt, facecolor=colors[i], alpha=.5)
        #axes[0][j].plot(actFreq, afLLmean, color=colors[i], label=tokens)

        #Yb, Yt = nbLLmean[k] - nbLLstd[k], nbLLmean[k] + nbLLstd[k]
        #axes[1][j].fill_between(nBlocks, Yb, Yt, facecolor=colors[i], alpha=.5)
        #axes[1][j].plot(nBlocks, nbLLmean, color=colors[i], label=tokens)            
    axes[0][0].set_ylabel(r'$\log \mathcal{P}(E_{LES} | E_{DNS})$')
    axes[1][0].set_ylabel(r'$\log \mathcal{P}(E_{LES} | E_{DNS})$')
    axes[0][0].set_ylim([-1100, -33])
    axes[1][0].set_ylim([-1100, -33])
    for j in range(nRes):
      axes[0][j].set_title(r'$Re_\lambda$ = %d' % REs[j])
      axes[0][j].set_xscale("symlog")
      axes[1][j].set_xscale("symlog")
      axes[0][j].set_yscale("symlog")
      axes[1][j].set_yscale("symlog")
      axes[0][j].grid()
      axes[1][j].grid()
      axes[0][j].set_xticks(actFreq)
      axes[1][j].set_xticks(nAgents)
      axes[0][j].set_xticklabels(actFreqLab)
      axes[1][j].set_xticklabels(nAgentsLab)
      #axes[j].set_xlim([0, 0.09])
      #axes[j].set_ylim([1e-4, 2e-1])
      axes[0][j].set_xlabel(r'$\tau_\eta / \Delta t_{RL}$')
      axes[1][j].set_xlabel(r'$N_{agents}$')
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    #assert(len(lines) == len(labels))
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
  p = argparse.ArgumentParser(description = "CSS plotter.")
  p.add_argument('--target', default='/users/novatig/smarties/apps/CUP3D_LES_HIT/target_RK_2blocks/', help="Path to target files directory")
  p.add_argument('--path', default='./', help="Simulation dira patb.")
  p.add_argument('--tokens', nargs='+', help="Text token distinguishing each series of runs")
  p.add_argument('--res', nargs='+', type=int, help="Reynolds numbers",
    default = [60, 70, 111, 151, 176, 190, 205])
  p.add_argument('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  args = p.parse_args()
  if args.labels is None: args.labels = args.tokens
  assert len(args.labels) == len(args.tokens)

  main(args.path, args.res, args.tokens, args.labels, args.target)

