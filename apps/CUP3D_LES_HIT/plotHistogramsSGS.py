#!/usr/bin/env python3
import argparse, matplotlib.pyplot as plt, numpy as np, os
from plot_eval_all_les import findBestHyperParams

colors = ['#377eb8', '#e41a1c', '#4daf4a', '#ff7f00', '#984ea3', '#ffff33', '#a65628', '#f781bf', '#999999']

RESSM = np.asarray([60, 65, 70, 76, 82, 88, 95, 103, 111, 120, 130, 140, 151, 163, 176, 190, 205])
CSSSM = [.2, .2075, .215, .22, .225, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23, .23]

def plot(ax, dirname, i):
  fname = dirname + '/sgsAnalysis.raw'
  rl_fname = dirname + '/simulation_000_00000/run_00000000/sgsAnalysis.raw'
  dns_fname = dirname + '/dnsAnalysis.raw'
  if   os.path.isfile(fname)     : f = np.fromfile(fname,     dtype=np.float64)
  elif os.path.isfile(rl_fname)  : f = np.fromfile(rl_fname,  dtype=np.float64)
  elif os.path.isfile(dns_fname) : f = np.fromfile(dns_fname, dtype=np.float64)
  else : assert False, 'sgsAnalysis file not found'
  # handles deprecate file saving sizes:
  if f.size % 904 is 0 : 
    nBINS, SHIFT = 900, 4
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = 0, nSAMPS
    x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
  elif f.size % 200 is 0 : 
    nBINS, SHIFT = 200, 200
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = int(0.2 * nSAMPS), int(0.8 * nSAMPS)
    x = np.linspace(-0.1, 0.15, 200)
  else :
    nBINS, SHIFT = 90, 4
    nSAMPS = f.size // (nBINS + SHIFT)
    nSKIPb, nSKIPe = 0, nSAMPS
    x = (np.arange(nBINS) + 0.5)/nBINS * 0.09
  
  f = f.reshape([nSAMPS, nBINS + SHIFT])
  #P =  np.zeros(nBINS)
  #for i in range(nSAMPS):
  #  MCS, VCS = f[i,0], f[i,2]
  #  denom = 1.0 / np.sqrt(2*np.pi*VCS) / nSAMPS
  #  P += denom * np.exp( -(x-MCS)**2 / (2*VCS) )
  #plt.plot(x, P)
  #print(nBINS, integral)

  #  Yb = np.percentile(f[nSKIPb:nSKIPe, SHIFT:], 41, axis=0)
  #  Yt = np.percentile(f[nSKIPb:nSKIPe, SHIFT:], 90, axis=0)
  #  meanH = np.percentile(f[nSKIPb:nSKIPe, SHIFT:], 70, axis=0)

  meanH = np.mean(f[nSKIPb:nSKIPe, SHIFT:], axis=0)
  stdH  = np.std( f[nSKIPb:nSKIPe, SHIFT:], axis=0)
  Yb, Yt = meanH-stdH, meanH+stdH

  integral = sum(meanH * (x[1]-x[0]))
  # max/min here are out of y axis, eliminate bug with log plot at -inf
  meanH = np.maximum(meanH/integral, 1e-1)
  Yb, Yt = np.maximum(Yb/integral, 1e-1), np.maximum(Yt/integral, 1e-1)
  #print( sum(meanH * (x[1]-x[0]) ) ) # should be 1 + floaterr

  ax.fill_between(x[:-1], Yb[:-1], Yt[:-1], facecolor=colors[i], alpha=.5)
  line = ax.plot(x[:-1], meanH[:-1], color=colors[i], label=dirname[-5:])
  return line

def main(runspath, REs, tokens, labels):
    nRes = len(REs)
    axes, lines = [], []
    #sharey=True, 
    fig, axes = plt.subplots(1,nRes, figsize=[11.4, 2.2], frameon=False, squeeze=True)

    for j in range(nRes):
        RE = REs[j]
        csssm = CSSSM[np.where(RESSM == RE)[0][0]]
        axes[j].plot([csssm ** 2, csssm ** 2], [0.2, 240], color=colors[4])
        
        for i in range(len(tokens)):
            dirn = findBestHyperParams(runspath, RE, tokens[i])
            l = plot(axes[j], dirn, i)
            if j == 0: lines += [l]

    axes[0].set_ylabel(r'$\mathcal{P}\left[\,C_s^2\right]$')
    #axes[1][0].invert_yaxis()
    #for j in range(1, nRes): axes[j].set_yticklabels([])
    for j in range(nRes):
      #axes[j].set_title(r'$Re_\lambda$ = %d' % REs[j])
      axes[j].set_yscale("log")
      axes[j].set_xlim([-1e-3, 0.09])
      axes[j].set_xticks([0.01, 0.04, 0.07])
      #axes[j].set_ylim([0.2, 225])
      axes[j].set_ylim([0.25, 150])
      axes[j].grid()
      axes[j].set_xlabel(r'$C_s^2$')
    #axes[0].legend(lines, labels, bbox_to_anchor=(-0.1, 2.5), borderaxespad=0)
    assert(len(lines) == len(labels))
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
  p.add_argument('tokens', nargs='+', help="Text token distinguishing each series of runs")
  p.add_argument('--path', default='./', help="Simulation dira patb.")
  p.add_argument('--res', nargs='+', type=int, help="Reynolds numbers",
    default = [60, 82, 111, 151, 190, 205])
  p.add_argument('--labels', nargs='+', help="Plot labels to assiciate to tokens")
  args = p.parse_args()
  if args.labels is None: args.labels = args.tokens
  assert len(args.labels) == len(args.tokens)

  main(args.path, args.res, args.tokens, args.labels)
