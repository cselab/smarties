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
# python plot_obs.py /path/to/folder ($2) ($3) ($4)
# optional args:
# ($2) Fraction of sequences not plotted (to lighten script and plot)
# ie. If $2=10 we plot all observed states for one sequence every 10 sequences.
# ($3) Agent id. Useful when running multiple agents.
# ($4) Master id. Useful when there are multiple master ranks.
#
# structure of .raw files is:
# [0/1/2] [state_cnter] [state] [action] [reward] [policy]
# (second column is 1 for first observation of an episode, 2 for last)

import argparse, sys, time, numpy as np, os, matplotlib.pyplot as plt

def nameAxis(colID, NS, NA, IREW, NCOL):
    assert(colID < NCOL)
    if colID<=0: return 'time steps'
    if colID==1: return 'agent status' #0 for init, 2 term, 3 trunc, 2 intermediate
    if colID==2: return 'time in episode'
    if colID<3+NS: return 'state component %d' % (colID-3)
    if colID<IREW: return 'action component %d' % (colID-3-NS)
    if colID==IREW: return 'intantaneous reward'
    return 'policy statistics component %d' % (colID-1-IREW)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = "Smarties state/action tuples plotting tool.")
  parser.add_argument('folder',
    help="Directoriy containing the observations file to plot.")
  parser.add_argument('plotCol', type=int, default=-1, nargs='?',
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
  parser.add_argument('--xAxisVar', type=int, default=-1,
    help="Allows plotting variables on the x-axis other than the time step counter.")
  parsed = parser.parse_args()


  sizes = np.fromfile(parsed.folder+'/problem_size.log', sep=' ')
  NS, NA, NP = int(sizes[0]), int(sizes[1]), int(sizes[2])
  NL=(NA*NA+NA)//2
  IREW=3+NS+NA
  NCOL=4+NS+NA+NP
  if parsed.plotCol < 0:
    print('States begin at col 3, actions at col ' + str(3+NS) + \
          ', rewards is col ' + str(IREW) + ', policy at col ' + \
          str(1+IREW) + ' end at ' + str(NCOL) + '.')
    ICOL = int( input("Column to print? ") )
  else:
    ICOL = parsed.plotCol

  FILE = "%s/agent%03d_rank%02d_obs.raw" \
         % (parsed.folder, parsed.agentID, parsed.workerRank)
  if os.path.isfile(FILE) is False:
    FILE = "%s/agent%03d_rank%02d_obs.raw" % (parsed.folder, parsed.agentID, 1)
  assert os.path.isfile(FILE), \
         "unable to find file %s" % FILE

  DATA = np.fromfile(FILE, dtype=np.float32)
  NROW = DATA.size // NCOL
  DATA = DATA.reshape(NROW, NCOL)

  terminals = np.argwhere(DATA[:,1]>=2.0).flatten()
  initials  = np.argwhere(DATA[:,1]< 1.0).flatten()
  print('Number of finished episodes %d, number of begun episodes %d' \
        % (len(terminals), len(initials)))
  print("Plot column %d out of %d. Log contains %d time steps from %d episodes." \
        % (ICOL, NCOL, NROW, len(terminals) ) )

  # ST = np.zeros(len(terminals)) # SAVE TERM STATES, TODO
  # for i in range(len(terminals)): ST[i] = DATA[terminals[i], ICOL]
  # np.savetxt('terminals.dat', ST, delimiter=',')

  #act = DATA[:,ICOL] # SCALE ACTIONS, TODO
  #max_a, min_a = 1.9, 0.1
  #DATA[:,ICOL] = min_a + 0.5 * (max_a-min_a) * (np.tanh(act) + 1);

  initialized = False
  for ind in range(0, len(terminals), parsed.nSkipPlot):
    init, term = initials[ind], terminals[ind]
    if parsed.nSkipInitT>0 and init<parsed.nSkipInitT: continue
    if parsed.nLastPlotT>0 and term>parsed.nLastPlotT: break

    print("Plotting episode starting from step %d to step %d." % (init,term) )
    span = range(init+1, term, 1)
    if parsed.xAxisVar >= 0:
      xes  = DATA[span, parsed.xAxisVar]
      xtrm = DATA[term, parsed.xAxisVar]
      xini = DATA[init, parsed.xAxisVar]
    else:
      xes, xtrm, xini = span, term, init

    if DATA[term,1] > 3: color='mo' # truncated state
    else: color='ro' # proper terminal state

    if initialized:
      plt.plot(xes,  DATA[span, ICOL], 'b-')
      plt.plot(xini, DATA[init, ICOL], 'go')
      plt.plot(xtrm, DATA[term, ICOL], color)
    else:
      initialized = True
      plt.plot(xes, DATA[span, ICOL], 'b-', label='trajectories')
      plt.plot(xini, DATA[init, ICOL], 'go', label='initial states')
      plt.plot(xtrm, DATA[term, ICOL], color, label='terminal states')

  plt.xlabel(nameAxis(parsed.xAxisVar, NS, NA, IREW, NCOL))
  plt.ylabel(nameAxis(ICOL, NS, NA, IREW, NCOL))
  plt.legend()
  plt.tight_layout()
  #plt.savefig('prova.png', dpi=100)
  plt.show()
