#!/usr/bin/env python3
import sys, os, glob
import numpy as np
import h5py as h5
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from matplotlib import cm

tmpFileName = 'tmpSF.h5'
lesGridSize = 32
dnsGridSize = 512
colors = ['#ff7f00', '#377eb8', '#4daf4a', '#984ea3', '#ffff33', '#a65628', '#f781bf', '#999999']
scale_DU = 10 * [None]

def areDnsIncrements(r_list):
    # either dns or les:
    assert(r_list.size == 5 or r_list.size > 7)
    return r_list.size > 7

def getStyle(lineid, rlist):
    if areDnsIncrements(rlist): return '-', '', 'k', 'k'
    else: return '', '.', colors[lineid], colors[lineid]
    #return ls, marker #

def getFilesList(simdir):
    filep = sorted(glob.glob(simdir + '/vel_*.h5'))
    if len(filep) == 0:
        rlpath = '/simulation_000_00000/run_00000000/vel_*.h5'
        filep = sorted(glob.glob(simdir + rlpath))
        assert(len(filep))
    return filep

def getNGrid(simdir):
    filep = getFilesList(simdir)[0]
    with h5.File(filep, 'r') as f:
        nGrid = np.array(f['data']).shape[0]
    return nGrid

def epsNuFromRe(Re, uEta = 1.0):
    C = 3.0 # np.sqrt(20.0/3)
    K = 2/3.0 * C * np.sqrt(15)
    eps = np.power(uEta*uEta * Re / K, 3.0/2.0)
    nu = np.power(uEta, 4) / eps
    return eps, nu

def getLogMeanLogStdev(sum1, sum2, N):
    mX = sum1 / N
    varX = sum2 / N - mX**2
    clipMx2 = np.maximum(mX**2, 1e-16)
    clipVarX = np.maximum(varX, 1e-16)
    mu = np.log(clipMx2 / np.sqrt(clipMx2 + clipVarX))
    sigma2 = np.log(1.0 + varX / clipMx2)
    return mu, np.sqrt(sigma2)

# this assumes log-normally distributed:
def getMeanUpperLower(sum1, sum2, N):
    mX = sum1 / N
    varX = sum2 / N - mX**2
    clipMx2 = np.maximum(mX**2, 1e-16)
    clipVarX = np.maximum(varX, 1e-16)
    mu = np.log(clipMx2 / np.sqrt(clipMx2 + clipVarX))
    sigma = np.sqrt( np.log(1.0 + varX / clipMx2) )
    return np.exp(mu), np.exp(mu+sigma), np.exp(mu-sigma)
'''
# this assumes normally distributed:
def getMeanUpperLower(sum1, sum2, N):
    mX = sum1 / N
    varX = np.maximum(sum2 / N - mX**2, 1e-16)
    lb = np.maximum(mX - np.sqrt(varX), 1e-16)
    return mX, lb, mX + np.sqrt(varX)
'''

def etaFromRe(Re, uEta = 1.0):
    eps, nu = epsNuFromRe(Re)
    eta = np.power(np.power(nu,3) / eps , 1.0/4)
    return eta

def etaEpsFromRe(Re, uEta = 1.0):
    eps, nu = epsNuFromRe(Re)
    eta = np.power(np.power(nu,3) / eps , 1.0/4)
    return eta, eps

def getVel(fPath):
    with h5.File(fPath, 'r') as f: data = np.array(f['data'])
    return data[:,:,:,0], data[:,:,:,1], data[:,:,:,2]

def realVelInc_fast(u,ax,r):
    nx, ny, nz = np.shape(u)
    ret = np.zeros((nx,ny,nz,2))
    # Roll array elements along a given axis. Elements that roll
    # beyond the last position are re-introduced at the first.
    ret[:,:,:,0] = np.roll(u,  int(r), axis=ax) - u
    ret[:,:,:,1] = np.roll(u, -int(r), axis=ax) - u
    return ret

def computeSF(fPath, nBins=200):
    u, v, w = getVel(fPath)
    u_rms = 1.0/3 * np.sqrt(np.mean(u**2 + v**2 + w**2))
    nx = u.shape[0]
    n_incr = int( np.log2(nx) )
    r_list = 2 ** np.arange(n_incr) #e.g. if nx=512, up to 256
    du_L_hist = np.zeros((2, nBins, n_incr))
    S_L_r = np.zeros((3, n_incr))
    bins = np.linspace(-30, 30, nBins+1)

    # print('Computing velocity increments:')
    # print('r (grid units):',r_list)
    for i, r in enumerate(r_list):
        # Longitudinal increment
        du = realVelInc_fast(u, ax=0, r=r).reshape(-1)
        dv = realVelInc_fast(v, ax=1, r=r).reshape(-1)
        dw = realVelInc_fast(w, ax=2, r=r).reshape(-1)
        incr = np.concatenate((du,dv,dw), axis=None) # / u_rms
        # Get the pdf of du^L(r)
        hist, edges = np.histogram(incr, bins=bins, density=False)
        # Center bins
        centers = 0.5*(edges[1:] + edges[:-1])
        dx = (edges[1:] - edges[:-1])
        # Store histogram
        du_L_hist[0,:,i] = centers
        du_L_hist[1,:,i] = hist
        # Compute/store the SF e.g. moments of PDF[du^L(r)]
        W = hist * dx
        S_L_r[0,i] = np.sum(W * np.fabs(centers)**2)/np.sum(W)
        S_L_r[1,i] = np.sum(W * np.fabs(centers)**3)/np.sum(W)
        S_L_r[2,i] = np.sum(W * np.fabs(centers)**4)/np.sum(W)

    # store grid spacing as if in dns resolution
    store_r_list = r_list * dnsGridSize / nx
    return store_r_list, du_L_hist, S_L_r

def saveSF_toHDF5(h5_path, h5_dir, data):
    r_list, du_L_hist, S_L_r = data
    print('Storing to', h5_dir)
    with h5.File(h5_path, 'a') as h5File:
        h5File.create_group(h5_dir)
        h5File.create_dataset(h5_dir+'/r_list', data=r_list)
        h5File.create_dataset(h5_dir+'/vel_incr', data=du_L_hist)
        h5File.create_dataset(h5_dir+'/SF', data=S_L_r)

def readStats_fromHDF5(SF_h5_path):
    with h5.File(SF_h5_path, 'r') as f:
        r_list    = np.array(f['r_list'])
        du_L_hist = np.array(f['vel_incr'])
        SF_L      = np.array(f['SF'])
    print('sizes:', r_list.shape, du_L_hist.shape, SF_L.shape)
    return r_list, du_L_hist, SF_L

def plot_vel_incr(h5path, ax, lineid, plot_rlist):
    with h5.File(h5path, 'r') as f:
      flist, N = list(f.keys()), len( list(f.keys()) )
      r_list = np.array(f[flist[0] + '/r_list'])
      ls, marker, c, _ = getStyle(lineid, r_list)
      du_sum1 = np.zeros_like(np.array(f[flist[0]+'/vel_incr']))
      du_sum2 = np.zeros_like(np.array(f[flist[0]+'/vel_incr']))
      #for snap in flist:
      for i in range(N//2, N):
          du_sum1 += np.array(f['/' + flist[i] + '/vel_incr'])
          du_sum2 += np.array(f['/' + flist[i] + '/vel_incr'])**2
      # centers do not actually vary in time:
      centers = du_sum1[0, :] / N
      dx = 0.5*(centers[1] - centers[0]) / N
      # norm = np.sum(du_sum1[1,:] / N * dx, axis=0)
      du_sum1, du_sum2 = du_sum1[1,:], du_sum2[1,:]
      dumean, duub, dulb = getMeanUpperLower(du_sum1, du_sum2, N)
      norm = np.sum(dumean * dx, axis=0)
      # normalize to get probabilities
      dumean, duub, dulb = dumean/norm, duub/norm, dulb/norm
      for i, r in enumerate(r_list):
        if not r in plot_rlist: continue
        #print(centers.shape)
        nc = dumean.shape[0]
        scale = (dumean[nc//2 -1, i] + dumean[nc//2, i])/2
        indscale = int(np.log2(r))
        global scale_DU
        if scale_DU[indscale] is None: scale_DU[indscale] = scale
        scale = 1 #scale_DU[indscale] / scale
        if areDnsIncrements(r_list):
          x = centers[:,i].flatten()
          m, lb, ub = dumean[:,i]*scale, duub[:,i]*scale, dulb[:,i]*scale
        else:
          #ins = np.arange(2, 200, 5)
          ins = np.arange(5, 200, 7)
          x, m = centers[ins,i].flatten(), dumean[ins,i]*scale
          lb, ub = duub[ins,i]*scale, dulb[ins,i]*scale
        ax.plot(x, m, color=c, ls=ls, marker=marker)
        if areDnsIncrements(r_list):
            ax.fill_between(x, lb, ub, color=c, alpha=0.3, linewidth=0)
        #x, y = centers, du_mean[:,i]
        #std2_y = du_std2[1,:,i]
        #idx = y>0
        #e = 0.434*np.sqrt(std2_y[idx]) / y[idx]
        #ax.plot(x[idx], y[idx], c=clist[i], \
        #        ls=ls, marker=marker, label='r={}'.format(r))
        #if areDnsIncrements(r_list):
        #  ax.fill_between()


def plot_SF(h5path, re, ax, lineid):
    with h5.File(h5path, 'r') as f:
      flist, N = list(f.keys()), len( list(f.keys()) )
      r_list  = np.array(f[flist[0] + '/r_list'])
      SF_sum1 = np.zeros_like(np.array(f[flist[0] + '/SF']))
      SF_sum2 = np.zeros_like(np.array(f[flist[0] + '/SF']))
      for snap in flist:
          SF_sum1 += np.array(f['/'+snap+'/SF'])
          SF_sum2 += np.array(f['/'+snap+'/SF'])**2
      #SF_mean, SF_std = getLogMeanLogStdev(SF_sum1, SF_sum2, N)
      SFmean, SFub, SFlb = getMeanUpperLower(SF_sum1, SF_sum2, N)
      # Symetric 1-sigma envelope in log space
      eta, eps = etaEpsFromRe(re)
      # r_list is stored as if increments in dns grid
      x = r_list * 2 * np.pi / dnsGridSize
      scale2 = np.power(eps * x, -2.0/3.0)
      scale3 = np.power(eps * x, -3.0/3.0)
      scale4 = np.power(eps * x, -4.0/3.0)
      ls, marker, _, c = getStyle(lineid, r_list)
      SF2 = SFmean[0,:] * scale2
      SF3 = SFmean[1,:] * scale3
      SF4 = SFmean[2,:] * scale4
      ax.plot(x/eta, SF2, c=c, ls=ls, marker=marker)
      #ax.plot(x/eta, SF3, c=c, ls=ls, marker=marker)
      #ax.plot(x/eta, SF4, c=c, ls=ls, marker=marker)
      if areDnsIncrements(r_list):
        SF2ub, SF2lb = SFub[0,:] * scale2, SFlb[0,:] * scale2
        SF3ub, SF3lb = SFub[1,:] * scale3, SFlb[1,:] * scale3
        SF4ub, SF4lb = SFub[2,:] * scale4, SFlb[2,:] * scale4
        ax.fill_between(x/eta, SF2ub,SF2lb, color=c, alpha=.3, linewidth=0) 
        #ax.fill_between(x/eta, SF3ub,SF3lb, color=c, alpha=.3, linewidth=0) 
        #ax.fill_between(x/eta, ub,lb, color=c, alpha=.3) 
      # Hand-fitted power laws
      #ax.plot(x, 0.32*(r_list)**(2.0/3), c='C0', ls='-', label='$r^{2/3}$')
      #ax.plot(x, 0.32*(r_list)**(3.0/3), c='C1', ls='-', label='$r^{3/3}$') 
      #ax.plot(x, 0.38*(r_list)**(4.0/3), c='C2', ls='-', label='$r^{4/3}$')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description = "Compute Structure Functions from DNS data.")

    parser.add_argument('simList', nargs='+',
      help="Path of simulations to process.")
    parser.set_defaults(simList=[])

    parser.add_argument('--Re', nargs='+', type=int,
      help="Reynolds number of simulations.")
    parser.set_defaults(Re=[60, 82, 111, 151, 190, 205])

    parser.add_argument('--plot_rlist',nargs='+',
      help="List of spatial increments to consider (in grid spacing units).")
    parser.set_defaults(plot_rlist=[32])

    parser.add_argument('--recompute', action='store_true',
      help="Only call the plotting routine.")
    parser.set_defaults(recompute=False)

    args = parser.parse_args()

    if args.recompute:
      for sim_t in args.simList:
        for re in args.Re:
            sim_dir = sim_t + '_RE%03d/' % re
            print("Processing simulation {}".format(sim_dir))
            fList = getFilesList(sim_dir)
            h5_path = sim_dir + '/' + tmpFileName
            if os.path.isfile(h5_path): os.remove(h5_path)

            for fPath in fList[10:]: #[::100]:
                print("Processing snapshot {}".format(fPath))
                h5_dir = fPath[-12:-3]
                print(h5_dir, sim_dir)
                data = computeSF(fPath)
                saveSF_toHDF5(h5_path, h5_dir, data)

    nSims = len(args.simList)

    simTypes = ['DNS', 'LES', 'RLLES']
    nTypes = len(simTypes)
    nPlots = len(args.Re) # nSims // nTypes
    #print(nPlots)
    #sharey='row', 
    fig, ax = plt.subplots(2,nPlots, figsize=[12.15, 4], frameon=False, squeeze=True)
    #fig, ax = plt.subplots(2, nPlots, figsize=(3*nPlots, 6))#, sharex=True, sharey=True)
    if nPlots==1: ax = np.expand_dims(ax, axis=1)

    for i, sim_t in enumerate(args.simList):
      for j, re in enumerate(args.Re):
        sim_dir = sim_t + '_RE%03d/' % re
        #plotID  = i // nTypes
        #simType = simTypes[i%nTypes]
        #ls, marker, clist_du, c_SN = getStyle(i, nGrid)
        #nGrid = getNGrid(sim_dir)
        h5_path = sim_dir + '/' + tmpFileName
        plot_vel_incr(h5_path, ax[0,j], i, args.plot_rlist)
        plot_SF(h5_path, re, ax[1,j], i)
        #ax[0,j].set_title(r'$Re_\lambda={}$'.format(re))
        ax[0,j].set_xlabel(r'$\delta u(\Delta) / u_{\eta}$')
        ax[1,j].set_xlabel(r'$r/\eta$')
        ax[1,j].set_xlim((0, None))
        ax[0,j].set_xlim((-12, 12))
        #ax[1,j].set_ylim((0.5, None))
        ax[0,j].set_ylim((1e-3, 99))
        ax[0,j].set_yscale('log')
        ax[0,j].grid(True)
        ax[1,j].grid(True)

    ax[0,0].set_ylabel(r'$\mathcal{P}\,\left[\delta u(\Delta)\right]$')
    #ax[0,0].legend()
    #ax[1,0].legend()
    ax[1,0].set_ylabel(r'$S^2(r) / (\epsilon r)^{2/3}$')
    
    
    #ax[1,0].set_yscale('log')
    plt.tight_layout()
    plt.show()
