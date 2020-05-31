#!/usr/bin/env python3.6
import re, argparse, numpy as np, glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from extractTargetFilesNonDim import getAllData
from extractTargetFilesNonDim import epsNuFromRe
from computeMeanIntegralQuantitiesNonDim import findAllParams
from computeMeanIntegralQuantitiesNonDim import readAllFiles
from computeSpectraNonDim import readAllSpectra

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def main_integral(targetpath, simdir, relambda, simnblocks):
    nSimBins = simnblocks * 16//2 - 1
    eps, nu = epsNuFromRe(relambda)
    runData = getAllData(simdir, eps, nu, nSimBins, 1)
    vecParams, vecMean, vecStd = readAllFiles(targetpath, [relambda])
    vecSpectra, vecEnStdev, fullSpectra, vecCovLogE = readAllSpectra(targetpath, [relambda])

    plt.figure()
    axes = [plt.subplot(1, 1, 1)]
    axes[0].set_xlabel(r'$k \eta$')
    axes[0].grid()
    axes[0].set_ylabel(r'$\Delta E(k) / E(k)$')

    ci = 0
    nyquist, nruns = vecSpectra.shape[0], vecSpectra.shape[1]
    print(nyquist)

    leta = np.power(nu**3 / eps, 0.25)
    Ekscal = np.power(nu**5 * eps, 0.25)
    nyquist = simnblocks * 16 // 2 - 1

    logE = np.log(runData['spectra'])
    logE = np.mean(logE, axis=0)
    logEtgt = vecSpectra[:nyquist, 0]
    print(logE.shape, logEtgt.shape, vecEnStdev.shape)
    dLogE = np.zeros(nyquist)
    for i in range(nyquist):
        dLogE[i] = (logE[i] - logEtgt[i]) / vecEnStdev[i]

    K = np.arange(1, nyquist+1, dtype=np.float64) * leta
    axes[0].plot(K, dLogE)
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targets', help="Directory containing the target files")
  parser.add_argument('--sim', help="Directory containing the dim to evaluate.")
  parser.add_argument('--re', type=int, help="Reynolds_lambda number of simulation.")
  parser.add_argument('--nblocks', type=int, help="Number of blocks per dim in simulation.")
  args = parser.parse_args()

  main_integral(args.targets, args.sim, args.re, args.nblocks)
