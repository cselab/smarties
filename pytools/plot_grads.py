#/usr/bin/env python
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
# SIMPLE PYTHON SCRIPT TO PLOT .raw GRADIENT FILES. Input is the path to the
# file, which is $(network_naem)_grads.raw (eg. net_grads.raw)

import sys
import numpy as np
import matplotlib.pyplot as plt
FILE=    sys.argv[1]


#np.savetxt(sys.stdout, np.fromfile(sys.argv[1], dtype='i4').reshape(2,10).transpose())
DATA = np.fromfile(FILE, dtype=np.float32)
NOUTS = int(DATA[0])
DATA = DATA[1:]
DATA = DATA.reshape(DATA.size // (2*NOUTS), 2*NOUTS)

EPS = 1e-5
for ind in range(0, NOUTS):
  plt.subplot(121)
  #plt.plot(DATA[:,ind]/DATA[:,NOUTS+ind], label=str(ind))
  #plt.semilogy(abs(DATA[:,ind]),label=str(ind))
  plt.semilogy(abs(DATA[:,ind])/(DATA[:,NOUTS+ind]+1e-16)+EPS, label=str(ind))
  plt.subplot(122)
  plt.semilogy(abs(DATA[:,NOUTS+ind])+EPS,'--',  label=str(ind))

plt.legend(loc=5, bbox_to_anchor=(1.2, 0.5))
#plt.savefig('prova.png', dpi=100)
plt.show()
