#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os, matplotlib.pyplot as plt

from computeMeanIntegralQuantitiesNonDim import findAllParams
from computeMeanIntegralQuantitiesNonDim import readAllFiles

from extractTargetFilesNonDim import epsNuFromRe
from extractTargetFilesNonDim import getAllData

from plot_eval_all_les import findBestHyperParams

colors = ['#e31a1c', '#1f78b4', '#33a02c', '#6a3d9a', '#ff7f00', '#b15928',
          '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def getSGSdissip(dirname, dissip_visc, nu):
    fname = dirname + '/sgsAnalysis.raw'
    rl_fname = dirname + '/simulation_000_00000/run_00000000/sgsAnalysis.raw'
    if   os.path.isfile(fname)    : f = np.fromfile(fname, dtype=np.float64)
    elif os.path.isfile(rl_fname) : f = np.fromfile(rl_fname, dtype=np.float64)
    else : assert False, 'sgsAnalysis file not found'
    if f.size % 904 is 0 : nBINS = 900
    else : nBINS = 90
    nSAMPS = f.size // (nBINS + 4)
    f = f.reshape([nSAMPS, nBINS + 4])
    dissip_SGS_coef = np.mean(f[:,1],axis=0) + np.mean(f[:,3],axis=0)
    return dissip_visc * (dissip_SGS_coef / nu)

def main_integral(runspath, target, tokens):
    REs = findAllParams(target)
    nRes, nBins = len(REs), 2 * 16//2 - 1
    vecParams, vecMean, vecStd = readAllFiles(target, REs)

    nQoItoPlot = len(QoI)
    fig, axes = plt.subplots(1,4, sharex=True, figsize=[12, 3], frameon=False, squeeze=True)

    for ax in axes.ravel(): ax.grid()
    #for ax in axes[:nQoItoPlot] : ax.set_xticklabels([])
    #for ax in axes[nQoItoPlot:] : ax.set_xlabel('Energy Injection Rate')
    nNonDimTkeMean,  nNonDimTkeStd  = np.zeros(nRes), np.zeros(nRes)
    nNonDimLintMean, nNonDimLintStd = np.zeros(nRes), np.zeros(nRes)
    nNonDimViscMean, nNonDimViscStd = np.zeros(nRes), np.zeros(nRes)
    nNonDimTotMean,  nNonDimTotStd  = np.zeros(nRes), np.zeros(nRes)
    LintMean = np.zeros(nRes)

    for k in range(nRes):
      for i in range(vecParams.shape[1]):
        eps, nu, re = vecParams[0, i], vecParams[1, i], vecParams[2, i]
        if np.abs(REs[k]-re) > 0 : continue
        eta   = np.power(nu*nu*nu/eps, 0.25)
        uprime = np.sqrt(2.0/3.0 * vecMean[1,i]);
        lambd = np.sqrt(15 * nu / eps) * uprime
        Kscal = np.power(eps, 2.0/3.0)
        nNonDimTkeMean[k]  = vecMean[1, i]/Kscal
        nNonDimTkeStd[k]   = vecStd [1, i]/Kscal
        LintMean[k] = vecMean[4, i]
        nNonDimLintMean[k] = vecMean[4, i] / eta
        nNonDimLintStd[k]  = vecStd[4, i] / eta
        nNonDimViscMean[k] = vecMean[6, i] / eps
        nNonDimViscStd[k]  = vecStd[6, i] / eps
        nNonDimTotMean[k]  = vecMean[7, i] / eps
        nNonDimTotStd[k]   = vecStd[7, i] / eps

    dataM = [nNonDimTkeMean, nNonDimLintMean, nNonDimViscMean, nNonDimTotMean]
    dataS = [nNonDimTkeStd,  nNonDimLintStd,  nNonDimViscStd,  nNonDimTotStd]
    for k in [0, 2]:
        top, bot = dataM[k] + dataS[k], dataM[k] - dataS[k]
        axes[k].fill_between(REs, bot, top, facecolor=colors[0], alpha=0.5)
        axes[k].plot(REs, dataM[k], 'o-', color=colors[0])

    for i in range(len(tokens)):
        for j in range(nRes):
            RE = REs[j]
            eps, nu = epsNuFromRe(REs[j])
            dirn = findBestHyperParams(runspath, RE, tokens[i])
            runData = getAllData(dirn, eps, nu, nBins, fSkip=1)
            dissip_SGS = getSGSdissip(dirn, runData['dissip_visc'], nu)
            eta   = np.power(nu*nu*nu/eps, 0.25)
            uprime = np.sqrt(2.0/3.0 * vecMean[1,i]);
            lambd = np.sqrt(15 * nu / eps) * uprime
            Kscal = np.power(eps, 2.0/3.0)
            nNonDimTkeMean[j]  = np.mean(runData['tke'])/Kscal
            nNonDimTkeStd[j]   = np.std(runData['tke'])/Kscal
            nNonDimLintMean[j] = np.mean(runData['l_integral']) / LintMean[j]
            nNonDimLintStd[j]  = np.std(runData['l_integral']) / LintMean[j]
            nNonDimViscMean[j] = np.mean(runData['dissip_visc']) / eps
            nNonDimViscStd[j]  = np.std(runData['dissip_visc']) / eps
            nNonDimTotMean[j]  = np.mean(dissip_SGS) / eps
            nNonDimTotStd[j]   = np.std(dissip_SGS) / eps
            #nNonDimTotMean[j]  = np.mean(runData['dissip_tot']) / eps
            #nNonDimTotStd[j]   = np.std(runData['dissip_tot']) / eps

        dataM = [nNonDimTkeMean, nNonDimLintMean, nNonDimViscMean, nNonDimTotMean]
        dataS = [nNonDimTkeStd,  nNonDimLintStd,  nNonDimViscStd,  nNonDimTotStd]
        for k in range(4):
            top, bot = dataM[k] + dataS[k], dataM[k] - dataS[k]
            axes[k].fill_between(REs, bot, top, facecolor=colors[i+1], alpha=0.5)
            axes[k].plot(REs, dataM[k], 'o-', color=colors[i+1])

    for ax in axes.ravel(): ax.set_xscale('log')
    for ax in axes.ravel(): ax.set_xlabel(r'$Re_\lambda$')

    axes[0].set_ylabel(r'$k \,/\, \epsilon^{2/3}$')
    axes[1].set_ylabel(r'$l_{int,\,LES} \,/\, l_{int,\,DNS}$')
    axes[2].set_ylabel(r'$\epsilon_{visc} \,/\, \epsilon$')
    axes[3].set_ylabel(r'$\epsilon_{SGS} \,/\, \epsilon$')

    #axes[1].set_yscale('log')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--target', help="Path to target files directory",
      default='/users/novatig/smarties/apps/CUP3D_LES_HIT/target_RKBPD24_2blocks/')
  parser.add_argument('--tokens', nargs='+', help="Text token distinguishing each series of runs")
  parser.add_argument('--runspath', default='./', help="Plot labels to assiciate to tokens")

  args = parser.parse_args()

  main_integral(args.runspath, args.target, args.tokens)
