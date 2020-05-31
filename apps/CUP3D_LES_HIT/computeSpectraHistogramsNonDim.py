#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
import matplotlib.pyplot as plt
from extractTargetFilesNonDim import gatherAllData
from extractTargetFilesNonDim import findAllParams
from extractTargetFilesNonDim import epsNuFromRe

colors_lines = ['#1f78b4', '#33a02c', '#ff7f00', '#e31a1c']
colors_hist = ['#a6cee3', '#b2df8a', '#fdbf6f', '#fb9a99']

nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]
doLogScale = True

def main_integral(path, nBlocks=32):
    REs = findAllParams(path)
    print(REs)
    nRes, nBins = len(REs), nBlocks * 16//2 - 1

    fig, axes = plt.subplots(1,8, sharey=False, figsize=[15, 3], frameon=False, squeeze=True)

    #axes[0].set_xlabel(r'$k \eta$')
    #axes[0].grid()
    #axes[0].set_ylabel(r'$\frac{E(k)}  {\eta u^2_\eta}$')
    ci = 0

    for i in range(1, nRes, 4):
        eps, nu = epsNuFromRe(REs[i])
        print(REs[i])
        data = gatherAllData(path, REs[i], eps, nu, nBins, fSkip=1)
        leta, lint = np.power(nu**3 / eps, 0.25), np.mean(data['l_integral'])
        Ekscal = np.power(nu**5 * eps, 0.25)
        ci += 1
        ck = 0
        for k in range(0,15,2):
            #print(data['spectra'].shape)
            Ek = data['spectra'][:, k] / Ekscal
            logEk = np.log(Ek)
            avgLogEk, stdLogEk = np.mean(logEk, axis=0), np.std(logEk, axis=0)
            minLogEk, maxLogEk = avgLogEk-3*stdLogEk, avgLogEk+3*stdLogEk
            minEk, maxEk = np.exp(minLogEk), np.exp(maxLogEk)
            color_l = colors_lines[ ci-1 ]
            color_h = colors_hist[ ci-1 ]

            if doLogScale:
                bins = np.linspace(minLogEk, maxLogEk, num=100)
                norml = 1 / (stdLogEk * np.sqrt(2*np.pi))
                P = np.exp(-0.5*((bins-avgLogEk)/stdLogEk)**2) * norml 
                axes[ck].plot(bins, P, '--',color=color_l)
                bins = np.linspace(minLogEk, maxLogEk, num=20)
                axes[ck].hist(logEk, bins=bins, density=True,
                              log=False, color=color_h, label=None)
            else:
                bins = np.geomspace(minEk, maxEk, num=20)
                axes[ck].hist(Ek, bins=bins, density=True,
                              log=True, color=color_h, label=None)
            #
            ck += 1
        '''
        
        Ekscal = np.power(nu**5 * eps, 0.25)
        K = np.arange(1, nyquist+1, dtype=np.float64)
        E = np.exp(vecSpectra[:,i])
        #X, Y = K, E / Ekscal
        X, Y = K * leta, E / Ekscal
        fit = np.array([EkBrief([k, eps,leta,lint,nu], popt) for k in K])/Ekscal
        Yb = np.exp(vecSpectra[:,i] - vecEnStdev[:,i])/Ekscal
        Yt = np.exp(vecSpectra[:,i] + vecEnStdev[:,i])/Ekscal

        fullE = np.exp(fullSpectra[:,i])
        fullN = fullSpectra.shape[0]
        fullK = np.arange(1, fullN+1, dtype=np.float64)
        eTot = np.sum(fullE)
        eCDF = np.array([np.sum(fullE[:k+1]) for k in range(fullN)]) / eTot
        #print(fullE, eCDF)

        label = r'$Re_\lambda=%d$' % re
        ci += 1
        color = colors[ - ci ]
        #axes[0].fill_between(X, Yb, Yt, facecolor=color, alpha=.5)
        #axes[0].plot(X[1:], fit[1:], 'o', color=color)
        #axes[0].plot(X, Y, color=color, label=label)
        #axes[1].plot(fullK, 1-eCDF, color=color, label=label)
        '''
    #for ax in axes: ax.set_xscale("log")
    for ax in axes: ax.set_yticks([])
    for ax in axes: ax.set_yticklabels([])
    k = 1
    for ax in axes:
        ax.set_title(r'$k = %d \pi / L$' % (2*k) )
        ax.set_xlabel(r'$\log\left[ E(k) \,/\, \eta u^2_\eta\right]$')
        k += 2
    axes[0].set_ylabel(r'$\mathcal{P}\left[\,\log\left( E \,/\, \eta u^2_\eta\right)\right]$')
    #axes[1].set_xlabel(r'$k$')
    #axes[1].set_ylabel(r'$1 - CDF(E)$')

    #axes[1].set_xlim([1,63])
    #axes[1].set_ylim([0.5, 1e-3])
    #axes[0].legend(loc='lower left', ncol=3)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('path',
    help="Simulation directory containing the 'Analysis' folder")
  args = parser.parse_args()

  main_integral(args.path)
