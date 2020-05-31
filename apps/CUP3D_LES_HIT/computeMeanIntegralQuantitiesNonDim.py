#!/usr/bin/env python3
import re, argparse, numpy as np, glob, os
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Velocity Gradient Stdev',
        'Integral Length Scale',
]

def findAllParams(fpath):
    REs = set()
    alldirs = glob.glob(fpath+'/scalars_*')
    for dirn in alldirs: REs.add(re.findall('RE\d\d\d',  dirn)[0][2:])
    REs = list(REs)
    print(REs)
    REs.sort()
    for i in range(len(REs)): REs[i] = float(REs[i])
    return REs

def readAllFiles(path, REs):
    nRes = len(REs)
    vecParams       = np.zeros([3, 0])
    vecMean, vecStd = np.zeros([nQoI, 0]), np.zeros([nQoI, 0])

    ind = 0
    for ei in range(nRes):
        fname = '%s/scalars_RE%03d' % (path, REs[ei])
        if( os.path.isfile(fname) == False) : continue
        vecParams = np.append(vecParams, np.zeros([3, 1]), axis=1)
        vecMean   = np.append(  vecMean, np.zeros([nQoI, 1]), axis=1)
        vecStd    = np.append(   vecStd, np.zeros([nQoI, 1]), axis=1)

        vecParams[2][ind] = REs[ei]
        file = open(fname,'r')

        line = file.readline().split()
        vecParams[0][ind] = float(line[1])
        assert (line[0] == 'eps')

        line = file.readline().split()
        vecParams[1][ind] = float(line[1])
        assert (line[0] == 'nu')

        line = file.readline().split()
        vecMean[0, ind], vecStd[0, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'dt')

        line = file.readline().split()
        vecMean[1, ind], vecStd[1, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'tKinEn')

        line = file.readline().split()
        vecMean[6, ind], vecStd[6, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'epsVis')

        line = file.readline().split()
        vecMean[7, ind], vecStd[7, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'epsTot')

        line = file.readline().split()
        vecMean[4, ind], vecStd[4, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'lInteg')

        line = file.readline().split()
        vecMean[5, ind], vecStd[5, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'tInteg')

        line = file.readline().split()
        vecMean[2, ind], vecStd[2, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'avg_Du')

        line = file.readline().split()
        vecMean[3, ind], vecStd[3, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'std_Du')

        file.close()
        ind = ind + 1

    # overwrite field 3 with h/lambda # std::sqrt(10 * nu * tke / dissip_visc)
    #coef = 10 * vecParams[1,:] / vecParams[0,:]
    #vecMean[2,:] = np.sqrt(coef * vecMean[1,:])
    #vecMean[2,:] = np.power(np.power(vecParams[1,:],3) / vecParams[0,:], 0.25)
    #vecMean[2,:] = np.power(vecParams[1,:] * vecParams[0,:], 0.25)
    #vecMean[2,:] = np.power(vecParams[1,:] / vecParams[0,:], 0.5)
    # Var F(x) ~ (F'(meanX))^2 Var x
    #vecStd [2,:] = 1 #vecStd[1,:] * (h * 0.5 * coef / (vecMean[2,:] ** 3) )

    return vecParams, vecMean, vecStd

def fitFunction(inps, dataM, dataV, row, func):
    if dataV is None :
      popt, pcov = curve_fit(func, inps, dataM[row,:])
    else:
      popt, pcov = curve_fit(func, inps, dataM[row,:], sigma = dataV[row,:])
    return popt

def main_integral(path):
    REs = findAllParams(path)
    nRes = len(REs)
    vecParams, vecMean, vecStd = readAllFiles(path, REs)

    def fitTKE(x, A,B,C):  return A * np.power(x[0], 2/3.0)
    def fitREL(x, A,B,C):  return A * np.power(x[0], 1/6.0) * np.power(x[1],-0.5)
    def fitTint(x, A,B,C): return A * np.power(x[0],-1/3.0) * np.power(x[1],1/6.0)
    def fitLint(x, A,B,C): return A * np.power(x[0],-1/24.0) * np.power(x[1], 1/12.0)
    def fitGrad(x, A,B,C): return A * np.power(x[0], 0.5) * np.power(x[1], -0.5)
    def fitFun(x, A,B,C):  return A * np.power(x[0], B) * np.power(x[1], C)

    nQoItoPlot = len(QoI)

    plt.figure()
    axes = []
    for i in range(2*nQoItoPlot) :
      axes = axes + [ plt.subplot(2, nQoItoPlot, i+1) ]
    for ax in axes: ax.grid()
    for ax in axes[:nQoItoPlot] : ax.set_xticklabels([])
    for ax in axes[nQoItoPlot:] : ax.set_xlabel('Energy Injection Rate')

    ni = 0
    for j in range(nQoItoPlot):
        funsMean = fitFun
        #if j == 0:  funsMean = fitFunDT
        #if j == 4:  funsMean = fitFunShift
        if j == 1:  funsMean = fitTKE
        funsStdv = fitFun
        pOptMean = fitFunction(vecParams, vecMean, vecStd, j, funsMean )
        pOptStdv = fitFunction(vecParams,  vecStd,   None, j, funsStdv )
        print('%s fit:' % QoI[j], pOptMean)
        axes[j].set_ylabel('%s' % QoI[j])

        E, M, S, fitM, fitS = [], [], [], [], []
        for k in range(nRes):
          for i in range(vecParams.shape[1]):
            eps, nu, re = vecParams[0, i], vecParams[1, i], vecParams[2, i]
            if np.abs(REs[k]-re) > 0 : continue
            E = E + [ REs[k] ]
            M = M + [ vecMean  [j, i] ]
            S = S + [ vecStd   [j, i] ]
            coefA, coefB, coefC = pOptMean[0], pOptMean[1], pOptMean[2]
            fitM += [ funsMean([eps, nu], coefA, coefB, coefC) ]
            coefA, coefB, coefC = pOptStdv[0], pOptStdv[1], pOptStdv[2]
            fitS += [ funsStdv([eps, nu], coefA, coefB, coefC) ]

        if len(E) is 0: assert(False)

        E, M, S = np.asarray(E), np.asarray(M), np.asarray(S)
        fitM, fitS = np.asarray(fitM), np.asarray(fitS)
        axes[j].fill_between(E, M-S, M+S, facecolor=colors[ni], alpha=0.5)
        axes[j].plot(E, M, color=colors[ni])
        axes[j].plot(E, fitM, 'o', color=colors[ni])
        axes[j].set_xscale('log')
        axes[j].set_yscale('log')

        axes[j+nQoItoPlot].plot(E, S, color=colors[ni])
        axes[j+nQoItoPlot].plot(E, fitS, 'o', color=colors[ni])
        axes[j+nQoItoPlot].set_xscale('log')
        axes[j+nQoItoPlot].set_yscale('log')
    #for ai in range(6):for ni in range(len(NUs)):for li in range(len(EXTs)):
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('--targets',
    help="Simulation directory containing the 'Analysis' folder")
  args = parser.parse_args()

  main_integral(args.targets)
