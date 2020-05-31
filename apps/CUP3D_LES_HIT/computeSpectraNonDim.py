#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from computeMeanIntegralQuantitiesNonDim import findAllParams
from computeMeanIntegralQuantitiesNonDim import readAllFiles
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
#colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
colors = ['#abd9e9', '#74add1', '#4575b4', '#313695', '#006837', '#1a9850',
          '#66bd63', '#a6d96a', '#d9ef8b', '#fee08b', '#fdae61', '#f46d43',
          '#d73027', '#a50026', '#8e0152', '#c51b7d', '#de77ae', '#f1b6da']
colors = ['#e9a3c9', '#c51b7d', '#d73027', '#fc8d59', '#fee08b', '#91cf60',
          '#1a9850', '#4575b4', '#91bfdb']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def EkFunc(x, C, CI, CE, BETA, P0):
    if(C   <1e-16): C   =1e-16
    #if(CI<0): CI=0
    if(CI  <1e-16): CI  =1e-16
    if(CE  <1e-16): CE  =1e-16
    if(BETA<1e-16): BETA=1e-16
    if(P0  <1e-16): P0  =1e-16
    CI, P0, BETA, CE = 0.001, 2, 5.4, 0.22
    k, eps, leta, lint, nu = x[0], x[1], x[2], x[3], x[4]
    #print(x.shape)
    #lint =  0.74885397 * np.power(eps, -0.0233311) * np.power(nu, 0.07192009)
    #leta = np.power(eps, -0.25) * np.power(nu, 0.75)
    #FL = np.power( k*lint / (np.abs(k*lint) + CI), 5/3.0 + P0 )
    FL = np.power( k*lint / np.sqrt((k*lint)**2 + CI), 5/3.0 + P0 )
    FE = np.exp( - BETA * ( np.power( (k*leta)**4 + CE**4, 0.25 ) - CE ) )
    ret = 2.7 * np.power(eps, 2/3.0) * np.power(k, -5/3.0) * FL * FE
    #print(C, CI, CE, BETA, P0)
    #print(eps[0], leta[:], lint[0], nu[0])
    return ret

def logEkFunc(x, C, CI, CE, BETA, P0):
    return np.log(EkFunc(x, C, CI, CE, BETA, P0))
def EkBrief(x, popt):
    return EkFunc(x, popt[0], popt[1], popt[2], popt[3], popt[4])

def readAllSpectra(path, REs):
    nRes = len(REs)
    allStdevs, allSpectra, fullSpectra, allCovLogE = None, None, None, None

    ind = 0
    for ei in range(nRes):
        ename = '%s/spectrumLogE_RE%03d' % (path, REs[ei])
        sname = '%s/stdevLogE_RE%03d' % (path, REs[ei])
        cname = '%s/invCovLogE_RE%03d' % (path, REs[ei])
        if os.path.isfile(ename) == False : continue
        if os.path.isfile(sname) == False : continue
        if os.path.isfile(cname) == False : continue
        modes, stdevs = np.loadtxt(ename, delimiter=',')[:,1], np.loadtxt(sname)
        invcov = np.loadtxt(cname, delimiter=', ')
        nyquist, fullSize = stdevs.size, modes.size
        if allSpectra is None :
            allStdevs, allSpectra = np.zeros([nyquist,0]), np.zeros([nyquist,0])
            fullSpectra = np.zeros([fullSize,0])
            #allCovLogE = np.zeros([nyquist, nyquist, 0])
        fullSpectra = np.append(fullSpectra, modes.reshape(fullSize,1), axis=1)
        modes = modes[:nyquist].reshape(nyquist,1)
        stdevs = stdevs.reshape(nyquist,1)
        allStdevs  = np.append(allStdevs,  stdevs, axis=1)
        allSpectra = np.append(allSpectra,  modes, axis=1)
        #allCovLogE = np.append(allCovLogE, invcov, axis=2)

    return allSpectra, allStdevs, fullSpectra, allCovLogE

def fitFunction(inps, dataM, dataV, row, func):
    if dataV is None :
      popt, pcov = curve_fit(func, inps, dataM[row,:])
    else:
      popt, pcov = curve_fit(func, inps, dataM[row,:], sigma = dataV[row,:])
    return popt

def fitSpectrum(vecParams, vecMean, vecSpectra, vecEnStdev):
    assert(vecSpectra.shape[1] == vecEnStdev.shape[1])
    assert(vecSpectra.shape[1] == vecParams.shape[1])
    assert(vecSpectra.shape[1] == vecMean.shape[1])
    nyquist, nruns = vecSpectra.shape[0], vecSpectra.shape[1]
    print(nyquist)
    kdata = np.zeros([nruns, nyquist, 5])
    for i in range(nruns):
        for j in range(nyquist):
            kdata[i, j, 0] = 0.5 + j
            kdata[i, j, 1] = vecParams[0,i]
            kdata[i, j, 2] = np.power(vecParams[1,i]**3 / vecParams[0,i], 0.25)
            kdata[i, j, 3] = vecMean[4,i]
            kdata[i, j, 4] = vecParams[1,i]
    #prepare vectors so that they are compatible with curve fit:
    ekdata, eksigma = vecSpectra.flatten(), vecEnStdev.flatten()
    kdata = kdata.reshape(nyquist*nruns, 5).transpose()
    bounds = [[ 1e-16,  1e-16,   1e-16,  1e-16,  1e-16],
              [np.inf, np.inf,  np.inf, np.inf, np.inf]]
    popt, pcov = curve_fit(logEkFunc, kdata, ekdata, sigma=eksigma,
        maxfev=100000, p0=[6.0, 1.0, 1.0, 5.24, 2.0], bounds=bounds)
    return popt, pcov

def main_integral(path):
    REs = findAllParams(path)
    nRes = len(REs)
    vecParams, vecMean, vecStd = readAllFiles(path, REs)
    vecSpectra, vecEnStdev, fullSpectra, _ = readAllSpectra(path, REs)
    popt, pcov = fitSpectrum(vecParams, vecMean, vecSpectra, vecEnStdev)
    C,CI,CE,BETA,P0 = popt[0], popt[1], popt[2], popt[3], popt[4]

    fig, axes = plt.subplots(1,2, figsize=[12, 3], frameon=False, squeeze=True)

    axes[0].set_xlabel(r'$k \eta$')
    axes[0].grid()
    axes[1].grid()
    axes[0].set_ylabel(r'$E(k) \,/\, \eta u^2_\eta$')

    ci = 0
    nyquist, nruns = vecSpectra.shape[0], vecSpectra.shape[1]
    print(popt, nyquist, fullSpectra.shape)

    for i in range(0, nruns, 2):
        eps, nu, re = vecParams[0,i], vecParams[1,i], vecParams[2,i]
        leta = np.power(vecParams[1,i]**3 / vecParams[0,i], 0.25)
        lint = vecMean[4,i]
        ri = np.argmin(np.abs(REs - re))
        #print(ri, i)

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
        axes[0].fill_between(X, Yb, Yt, facecolor=color, alpha=.5)
        #axes[0].plot(X[1:], fit[1:], 'o', color=color)
        axes[0].plot(X, Y, color=color, label=label)
        axes[1].plot(fullK, 1-eCDF, color=color, label=label)

    xTheory = np.zeros(2)
    xTheory[0], xTheory[1] = 0.025, 0.25
    yTheory = 6.41241 * np.power(xTheory, -5.0/3.0)
    axes[0].plot(xTheory, yTheory, 'k--')
    axes[1].plot([15, 15], [1, 1e-3], 'k')
    axes[1].set_xlabel(r'$k \cdot L \,/\, 2 \pi$')
    axes[1].set_ylabel(r'$1 - CDF\,\left[ E(k)\right]$')
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    #axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim([1,63])
    axes[1].set_ylim([0.5, 1e-3])
    axes[0].legend(loc='lower left', ncol=3)
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targets',
    help="Simulation directory containing the 'Analysis' folder")
  args = parser.parse_args()

  main_integral(args.targets)
