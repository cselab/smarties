#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt

from computeMeanIntegralQuantitiesNonDim import findAllParams
from computeMeanIntegralQuantitiesNonDim import readAllFiles

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def main_integral(path):
    REs = findAllParams(path)
    nRes = len(REs)
    vecParams, vecMean, vecStd = readAllFiles(path, REs)

    nQoItoPlot = len(QoI)
    fig, axes = plt.subplots(2,2, sharex=True, figsize=[6, 3], frameon=False, squeeze=True)

    for ax in axes.ravel(): ax.grid()
    #for ax in axes[:nQoItoPlot] : ax.set_xticklabels([])
    #for ax in axes[nQoItoPlot:] : ax.set_xlabel('Energy Injection Rate')
    nNonDimTkeMean,  nNonDimTkeStd  = np.zeros(nRes), np.zeros(nRes)
    nNonDimLintMean, nNonDimLintStd = np.zeros(nRes), np.zeros(nRes)
    nNonDimViscMean, nNonDimViscStd = np.zeros(nRes), np.zeros(nRes)
    nNonDimTotMean,  nNonDimTotStd  = np.zeros(nRes), np.zeros(nRes)

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
        nNonDimLintMean[k] = vecMean[4, i] / eta
        nNonDimLintStd[k]  = vecStd[4, i] / eta
        nNonDimViscMean[k] = vecMean[6, i] / eps
        nNonDimViscStd[k]  = vecStd[6, i] / eps
        nNonDimTotMean[k]  = vecMean[7, i] / eps
        nNonDimTotStd[k]   = vecStd[7, i] / eps

    dataM = [nNonDimTkeMean, nNonDimLintMean, nNonDimViscMean, nNonDimTotMean]
    dataS = [nNonDimTkeStd,  nNonDimLintStd,  nNonDimViscStd,  nNonDimTotStd]
    for k in range(4):
        i, j = k // 2, k % 2
        top, bot = dataM[k] + dataS[k], dataM[k] - dataS[k]
        axes[i][j].fill_between(REs, bot, top, facecolor=colors[0], alpha=0.5)
        axes[i][j].plot(REs, dataM[k], 'o-', color=colors[0])

    for ax in axes.ravel(): ax.set_xscale('log')
    axes[1][0].set_xlabel(r'$Re_\lambda$')
    axes[1][1].set_xlabel(r'$Re_\lambda$')

    axes[0][0].set_ylabel(r'$k / \epsilon^{3/2}$')
    axes[0][1].set_ylabel(r'$l_{int} / \eta$')
    axes[1][0].set_ylabel(r'$\epsilon_{visc} / \epsilon$')
    axes[1][1].set_ylabel(r'$\epsilon_{tot} / \epsilon$')

    axes[0][1].set_yscale('log')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targets',
    help="Simulation directory containing the 'Analysis' folder")
  args = parser.parse_args()

  main_integral(args.targets)
