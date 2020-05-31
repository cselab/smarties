#!/usr/bin/env python3
import re, argparse, numpy as np, glob
#from sklearn.neighbors.kde import KernelDensity
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6', '#ffff99']
nQoI = 8
h = 2 * np.pi / (16*16)
QoI = [ 'Time Step Size',
        'Turbulent Kinetic Energy',
        'Velocity Gradient',
        'Integral Length Scale'
]
        #'Velocity Gradient Stdev',

def findAllParams(path):
    NUs, EPSs = set(), set()
    alldirs = glob.glob(path+'/scalars_*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d\d', dirn)[0][2:])
    NUs, EPSs  = list( NUs), list(EPSs)
    NUs.sort()
    EPSs.sort()
    for i in range(len(NUs)): NUs[i] = float(NUs[i])
    for i in range(len(EPSs)): EPSs[i] = float(EPSs[i])
    return NUs, EPSs

def readAllFiles(path, NUs, EPSs):
    nNus, nEps, nTot = len(NUs), len(EPSs), len(NUs) * len(EPSs)
    vecParams       = np.zeros([2, nTot])
    vecMean, vecStd = np.zeros([nQoI, nTot]), np.zeros([nQoI, nTot])

    ind = 0
    for ei in range(nEps):
      for ni in range(nNus):
        vecParams[0, ind] = EPSs[ei]
        vecParams[1, ind] =  NUs[ni]
        file = open('%s/scalars_EPS%.03f_NU%.04f'%(path, EPSs[ei], NUs[ni]),'r')

        line = file.readline().split() # skip
        assert (line[0] == 'eps' and np.abs( float(line[1]) - EPSs[ei] ) < 1e-8)

        line = file.readline().split() # skip
        assert (line[0] == 'nu'  and np.abs( float(line[1]) -  NUs[ni] ) < 1e-8)

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
        vecMean[3, ind], vecStd[3, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'lInteg')

        line = file.readline().split()
        vecMean[5, ind], vecStd[5, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'tInteg')

        line = file.readline().split()
        vecMean[2, ind], vecStd[2, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'avg_Du')

        line = file.readline().split()
        vecMean[4, ind], vecStd[4, ind] = float(line[1]), float(line[2])
        assert (line[0] == 'std_Du')

        line = file.readline().split()
        assert (line[0] == 'ReLamd')
        #if float(line[1])<120 and float(line[1])>60 : print(vecParams[:,ind])
        
        file.close()
        ind = ind + 1

    # overwrite field 3 with h/lambda # std::sqrt(10 * nu * tke / dissip_visc)
    coef = 10 * vecParams[1,:] / vecParams[0,:]
    vecMean[2,:] = np.sqrt(coef * vecMean[1,:])
    # Var F(x) ~ (F'(meanX))^2 Var x
    vecStd [2,:] = vecStd[1,:] * (h * 0.5 * coef / (vecMean[2,:] ** 3) )

    # overwrite field 4 with dt * ( eps / tke )
    #vecMean[5,:] = (vecMean[7,:] - vecMean[6,:]) / vecMean[7,:]
    #vecStd[5,:] = vecStd[6,:]
    # Var F(x) ~ (F'(meanX))^2 Var x
    #vecStd [4,:] = vecStd[2,:] * (h * 0.5 * coef / (vecMean[3,:] ** 3) ) ** 2
    
    return vecParams, vecMean, vecStd

def fitFunction(inps, dataM, dataV, row, func):
    if dataV is None : popt, pcov = curve_fit(func, inps, dataM[row,:])
    else:
      popt, pcov = curve_fit(func, inps, dataM[row,:], sigma = dataV[row,:])
    return popt

def main_integral(path):
    NUs, EPSs = findAllParams(path)
    nNus, nEps = len(NUs), len(EPSs)
    vecParams, vecMean, vecStd = readAllFiles(path, NUs, EPSs)

    def fitTKE(x, A,B,C):  return A * np.power(x[0], 2/3.0)
    def fitREL(x, A,B,C):  return A * np.power(x[0], 1/6.0) * np.power(x[1],-0.5)
    def fitTint(x, A,B,C): return A * np.power(x[0],-1/3.0) * np.power(x[1],1/6.0)
    def fitLint(x, A,B,C): return A * np.power(x[0],-1/24.0) * np.power(x[1], 1/12.0)
    def fitGrad(x, A,B,C): return A * np.power(x[0], 0.5) * np.power(x[1], -0.5)
    def fitFun(x, A,B,C):  return A * np.power(x[0], B) * np.power(x[1], C)
    def fitFunShift(x, A,B,C,D):  return B + A * np.power(x[0], -1/16.0) * np.power(x[1], 1/6.0)
    def fitFunDT(x, A,B,C):  return A * np.power(x[0], -1/3.0) * np.power(x[1], 1/8.0)

    nQoItoPlot = len(QoI)

    plt.figure()
    axes = []
    for i in range(2*nQoItoPlot) :
      axes = axes + [ plt.subplot(2, nQoItoPlot, i+1) ]
    for ax in axes: ax.grid()
    for ax in axes[:nQoItoPlot] : ax.set_xticklabels([])
    for ax in axes[nQoItoPlot:] : ax.set_xlabel('Energy Injection Rate')

    for j in range(nQoItoPlot):
      funsMean = fitFun
      #if j == 0:  funsMean = fitFunDT
      if j == 4:  funsMean = fitFunShift
      if j == 1:  funsMean = fitTKE
      funsStdv = fitFun
      pOptMean = fitFunction(vecParams, vecMean, vecStd, j, funsMean )
      pOptStdv = fitFunction(vecParams,  vecStd,   None, j, funsStdv )
      print('%s fit:' % QoI[j], pOptMean)
      axes[j].set_ylabel('%s' % QoI[j])

      for ni in range(nNus):
        E, M, S, fitM, fitS = [], [], [], [], []
        for ei in range(nEps):
          for i in range(vecParams.shape[1]):
            eps, nu = vecParams[0, i], vecParams[1, i]
            if np.abs(EPSs[ei]-eps) > 0 or np.abs( NUs[ni]- nu) > 0 : continue
            E = E + [ eps ]
            M = M + [ vecMean[j, i] ]
            S = S + [ vecStd [j, i] ]
            if len(pOptMean) == 3:
              coefA, coefB, coefC = pOptMean[0], pOptMean[1], pOptMean[2]
              fitM += [funsMean([eps, nu], coefA, coefB, coefC) ]
            else:
             cA, cB, cC, cD = pOptMean[0], pOptMean[1], pOptMean[2], pOptMean[3]
             fitM += [funsMean([eps, nu], cA, cB, cC, cD) ]
            coefA, coefB, coefC = pOptStdv[0], pOptStdv[1], pOptStdv[2]
            fitS += [funsStdv([eps, nu], coefA, coefB, coefC) ]

        if len(E) is 0:
          print('emnpty NU value')
          continue
        E, M, S = np.asarray(E), np.asarray(M), np.asarray(S)
        fitM, fitS = np.asarray(fitM), np.asarray(fitS)
        axes[j].fill_between(E, M-S, M+S, facecolor=colors[ni], alpha=0.5)
        axes[j].plot(E, M, color=colors[ni], label='nu=%f'%NUs[ni])
        axes[j].plot(E, fitM, 'o', color=colors[ni])
        axes[j].set_xscale('log')
        axes[j].set_yscale('log')
        axes[j].legend()
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
