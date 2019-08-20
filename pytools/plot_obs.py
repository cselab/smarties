#/usr/bin/env python
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

import sys
import numpy as np
import matplotlib.pyplot as plt
PATH=    sys.argv[1]
sizes = np.fromfile(PATH+'/problem_size.log', sep=' ')
NS, NA, NP = int(sizes[0]), int(sizes[1]), int(sizes[2])
NL=(NA*NA+NA)//2
IREW=3+NS+NA
NCOL=4+NS+NA+NP
print('States begin at col 3, actions at col '+str(3+NS)+', rewards is col '+str(IREW)+', policy at col '+str(1+IREW)+' end at '+str(NCOL)+'.')
ICOL = int( input("Column to print? ") )
#COLMAX = 1e7
#COLMAX = 8e6
#COLMAX = 1e6
COLMAX = -1

if len(sys.argv) > 2: SKIP=int(sys.argv[2])
else: SKIP = 10

if len(sys.argv) > 3: AGENTID=int(sys.argv[3])
else: AGENTID = 0

if len(sys.argv) > 4: RANK=int(sys.argv[4])
else: RANK = 0

if len(sys.argv) > 5: XAXIS=int(sys.argv[5])
else: XAXIS = -1

if len(sys.argv) > 6: IND0=int(sys.argv[6])
else: IND0 = 0

FILE = "%s/agent%03d_rank%02d_obs.raw" % (PATH, AGENTID, RANK)
#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NROW = DATA.size // NCOL
DATA = DATA.reshape(NROW, NCOL)

terminals = np.argwhere(DATA[:,1]>=2.)
initials  = np.argwhere(abs(DATA[:,1]-0.1)<0.1)
print('size of terminals %d, size of initials %d' % (len(terminals), len(initials)))

print("Plot column %d out of %d. Log contains %d time steps from %d episodes." \
      % (ICOL, NCOL, NROW, len(terminals) ) )
inds = np.arange(0,NROW)
ST = np.zeros(len(terminals))

act = DATA[:,ICOL]
max_a, min_a = 1.9, 0.1
#DATA[:,ICOL] = min_a + 0.5 * (max_a-min_a) * (np.tanh(act) + 1);
for ind in range(IND0, len(terminals), SKIP):
  term = terminals[ind]; term = term[0]
  init =  initials[ind]; init = init[0]
  if COLMAX>0 and term>COLMAX: break;
  span = range(init+1, term, 1)
  print("Plotting episode starting from step %d to step %d." % (init,term) )
  if XAXIS>=0:
    xes, xtrm, xini = DATA[span,XAXIS], DATA[term,XAXIS], DATA[init,XAXIS]
  else:
    xes, xtrm, xini = inds[span]      , inds[term],       inds[init]
  #print(xini, xes, xtrm)
  if (ind % 1) == 0:
    if ind==IND0:
      plt.plot(xes, DATA[span,ICOL], 'bo', label='x-trajectory')
    else:
      plt.plot(xes, DATA[span,ICOL], 'bo')

  #plt.plot(inds, DATA[:,ICOL])
  ST[ind] = DATA[term, ICOL]

  if ind==IND0:
    plt.plot(xini, DATA[init, ICOL], 'go', label='terminal x')
  else:
    plt.plot(xini, DATA[init, ICOL], 'go')
  if DATA[term,0] > 3: color='mo'
  else: color='ro'
  if ind==IND0:
    plt.plot(xtrm, DATA[term, ICOL], color, label='terminal x')
  else:
    plt.plot(xtrm, DATA[term, ICOL], color)
#plt.legend(loc=4)
#plt.ylabel('x',fontsize=16)
#plt.xlabel('t',fontsize=16)
#if COLMAX>0:plt.axis([0, COLMAX, -50, 150])
#plt.semilogy(inds, 1/np.sqrt(DATA[:,ICOL]))
plt.tight_layout()
#plt.semilogy(inds[terminals], 1/np.sqrt(DATA[terminals-1,ICOL]), 'ro')
np.savetxt('terminals.dat', ST, delimiter=',')
#plt.savefig('prova.png', dpi=100)
plt.show()
