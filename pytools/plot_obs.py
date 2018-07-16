#/usr/bin/env python
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
# SIMPLE PYTHON SCRIPT TO PLOT .raw AGENT OBS FILES
#
# usage:
# python python_plot_obs.py len_state_vec len_action_vec path/to/file.raw column_ID_to_plot
# also, optional: ( number_of_elements_in_policy_vector )
# otherwise, assumed continuous pol with 2*NA components (mean, precision of gaussian)
#
# structure of .raw files is:
# transition_id [0/1/2] [state] [action] [reward] [policy]
# (second column is 1 for first observation of an episode, 2 for last)

import sys
import numpy as np
import matplotlib.pyplot as plt
PATH=    sys.argv[1]
sizes = np.fromfile(PATH+'/problem_size.log', sep=' ')
NS, NA, NP = int(sizes[0]), int(sizes[1]), int(sizes[2])
NL=(NA*NA+NA)//2
NREW=3+NS+NA
NCOL=3+NS+NA+NP
print('States begin at 2, actions at '+str(2+NS)+' policy at '+str(3+NS+NA)+' end at '+str(NCOL)+'.')
ICOL = int( input("Column to print? ") )
#COLMAX = 1e7
#COLMAX = 8e6
#COLMAX = 1e6
COLMAX = -1

if len(sys.argv) > 2: SKIP=int(sys.argv[2])
else: SKIP = 10

if len(sys.argv) > 3: XAXIS=int(sys.argv[3])
else: XAXIS = -1

if len(sys.argv) > 4: RANK=int(sys.argv[4])
else: RANK = 0

if len(sys.argv) > 5: AGENTID=int(sys.argv[5])
else: AGENTID = 0

if len(sys.argv) > 6: IND0=int(sys.argv[6])
else: IND0 = 0

FILE = "%s/obs_rank%02d_agent%03d.raw" % (PATH, RANK, AGENTID)
#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NROW = DATA.size // NCOL
DATA = DATA.reshape(NROW, NCOL)

terminals = np.argwhere(DATA[:,0]>=2.)
initials  = np.argwhere(abs(DATA[:,0]-1.1)<0.1)
print(np.mean(DATA[terminals,NREW-1]))
print(np.std(DATA[:,NREW-1]), np.mean(DATA[:,NREW-1]))
print(NROW, NCOL,ICOL,len(terminals))
inds = np.arange(0,NROW)
ST = np.zeros(len(terminals))

for ind in range(IND0, len(terminals), SKIP):
  term = terminals[ind]; term = term[0]
  init =  initials[ind]; init = init[0]
  if COLMAX>0 and term>COLMAX: break;
  span = range(init+1, term, 1)
  print(init,term)
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
  #if ICOL >= NREW: plottrm = term-1
  #else: plottrm = term

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
