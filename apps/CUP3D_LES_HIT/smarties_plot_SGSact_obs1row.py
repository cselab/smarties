#!/usr/bin/env python3
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
# SIMPLE PYTHON SCRIPT TO PLOT .raw AGENT OBS FILES
#
# usage:
# python smarties_plot_obs.py /path/to/folder ($2) ($3) ($4)
# optional args:
# ($2) Fraction of sequences not plotted (to lighten script and plot)
# ie. If $2=10 we plot all observed states for one sequence every 10 sequences.
# ($3) Agent id. Useful when running multiple agents.
# ($4) Master id. Useful when there are multiple master ranks.
#
'''
smarties_plot_SGSact_obs.py 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN16_032PD_RE065 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN16_032PD_RE088 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN16_032PD_RE120 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN16_032PD_RE163 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN17_032PD_RE065 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN17_032PD_RE088 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN17_032PD_RE120 
BlockAgents_ALLINP_FFNN_${NB}blocks_act${AF}_sim16_RUN17_032PD_RE163 --plotCol 34 --nSkipPlot 10 --nSkipInitT 0 --xAxisVar 4 9 15 33 16 17
'''

import argparse, sys, time, numpy as np, os, matplotlib.pyplot as plt

colors = ['#1f78b4', '#33a02c', '#ff7f00', '#e31a1c', '#a6cee3', '#b2df8a', '#fdbf6f', '#fb9a99']
initialized = False

def nameAxis(colID, NS, NA, IREW, NCOL):
    assert(colID < NCOL)
    if colID<=0: return 'time steps'
    if colID==1: return 'agent status' #0 for init, 2 term, 3 trunc, 2 intermediate
    if colID==2: return 'time in episode'
    if colID<3+NS: return 'state component %d' % (colID-3)
    if colID<IREW: return 'action component %d' % (colID-3-NS)
    if colID==IREW: return 'intantaneous reward'
    return 'policy statistics component %d' % (colID-1-IREW)

def plotFile(ax, xAxisVar, parsed, dirn, ICOL, NCOL, NS, NA, IREW, color):
  FILE = "%s/agent%03d_rank%02d_obs.raw" % (dirn, parsed.agentID, parsed.workerRank)

  if os.path.isfile(FILE) is False: # check if rank 1 wrote smth to file:
    FILE = "%s/agent%03d_rank%02d_obs.raw" % (dirn, parsed.agentID, 1)

  assert os.path.isfile(FILE), "unable to find file %s" % FILE

  DATA = np.fromfile(FILE, dtype=np.float32)
  NROW = DATA.size // NCOL
  DATA = DATA.reshape(NROW, NCOL)

  terminals = np.argwhere(DATA[:,1]>=2.0).flatten()
  initials  = np.argwhere(DATA[:,1]< 1.0).flatten()
  print('Number of finished episodes %d, number of begun episodes %d' \
        % (len(terminals), len(initials)))
  print("Plot column %d out of %d. Log contains %d time steps from %d episodes." \
        % (ICOL, NCOL, NROW, len(terminals) ) )

  # do not plot actions and policy values for terminal states:
  xAndyNotPol = ( xAxisVar < 3+NS or xAxisVar == IREW) and \
                ( ICOL <= 3+NS or ICOL == IREW )

  for ind in range(0, len(terminals), parsed.nSkipPlot):
    init, term = initials[ind], terminals[ind]
    if parsed.nSkipInitT>0 and init<parsed.nSkipInitT: continue
    if parsed.nLastPlotT>0 and term>parsed.nLastPlotT: break
    span = range(init, term + xAndyNotPol, 10)

    print("Plotting episode starting from step %d to step %d." % (init,term) )
    if xAxisVar >= 0:
      ax.plot(DATA[span, xAxisVar],  DATA[span, ICOL], '.', color=color, ms=1)
    else:
      ax.plot(span,  DATA[span, ICOL], '-', color=color)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Smarties state/action tuples plotting tool.")
  parser.add_argument('folders', nargs='+',
    help="Directories containing the observations file to plot.")
  parser.add_argument('--plotCol', type=int, default=31,
    help="Variable to plot. If unset it will be asked interactively.")
  parser.add_argument('--nSkipPlot', type=int, default=10,
    help="Determines how many episode's trajectories are skipped for one that is visualized in the figure.")
  parser.add_argument('--nSkipInitT', type=int, default=0,
    help="Allows skipping the first #arg time steps from the figure.")
  parser.add_argument('--nLastPlotT', type=int, default=-1,
    help="Allows plotting only the first #arg time steps contained in the file.")
  parser.add_argument('--workerRank', type=int, default=0,
    help="Will plot the trajectories obtained by this MPI rank. "
         "Defaults to 0 in most cases, when the main learner rank obtains all data. "
         "If run had MPI ranks collecting episodes it will default to 1.")
  parser.add_argument('--agentID', type=int, default=0,
    help="Whose agent's trajectories will be plotted. Defaults to 0 and only affects multi-agent environments.")
  parser.add_argument('--xAxisVar', type=int, default=[-1], nargs='+',
    help="Allows plotting variables on the x-axis other than the time step counter.")
  parsed = parser.parse_args()

  sizes = np.fromfile(parsed.folders[0]+'/problem_size.log', sep=' ')
  NS, NA, NP = int(sizes[0]), int(sizes[1]), int(sizes[2])
  NL, IREW, NCOL = (NA*NA+NA)//2, 3+NS+NA, 4+NS+NA+NP

  parsed.xAxisVar = [3, 8, 17, 30, 14, 15]
  nFigs = len(parsed.xAxisVar)
  fig, axes = plt.subplots(1, nFigs, sharey=True, figsize=[12, 3], frameon=False, squeeze=True)

  if parsed.plotCol < 0:
    print('States begin at col 3, actions at col ' + str(3+NS) + \
          ', rewards is col ' + str(IREW) + ', policy at col ' + \
          str(1+IREW) + ' last at ' + str(NCOL-1) + '.')
    ICOL = int( input("Column to print? ") )
  else: ICOL = parsed.plotCol

  for ix in range(len(parsed.xAxisVar)):
    xAxisVar = parsed.xAxisVar[ix]

    colorid = 0
    for dirn in parsed.folders:
      color, colorid = colors[colorid], colorid + 1
      plotFile(axes[ix], xAxisVar, parsed, dirn, ICOL, NCOL, NS, NA, IREW, color)

    #axes[ix].set_xlabel(nameAxis(xAxisVar, NS, NA, IREW, NCOL))

  axes[0].set_xlabel(r'$\lambda_1^{\nabla u} \, \cdot \, K / \epsilon$')
  axes[1].set_xlabel(r'$\lambda_1^{\Delta u} \, \cdot \, \eta \cdot K / \epsilon$')
  axes[2].set_xlabel(r'$E(k = 4 \pi / L) \, / \, u_\eta^2$')
  axes[3].set_xlabel(r'$E(k = 30 \pi / L) \, / \, u_\eta^2$')
  axes[4].set_xlabel(r'$\epsilon_{visc} \, / \, \epsilon$')
  axes[5].set_xlabel(r'$\epsilon_{tot} \, / \, \epsilon$')
  axes[0].set_ylabel(r'$C_s^2$')

  #axes[0].set_ylabel(nameAxis(ICOL, NS, NA, IREW, NCOL))
  axes[0].set_yscale("log")
  axes[0].set_ylim([0.018, 0.058])

  #axes[0].set_xlim([1, 38])
  axes[0].set_xlim([0.1, 7.5])
  axes[1].set_xlim([-6.5, 6.5])
  axes[2].set_xlim([4.5, 25])
  axes[3].set_xlim([0.005, 1])
  axes[4].set_xlim([0.1, 0.60])
  axes[5].set_xlim([0.75, 1.25])

  #plt.legend()
  fig.tight_layout()
  #plt.savefig('prova.png', dpi=100)
  plt.show()
