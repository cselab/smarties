#/usr/bin/env python
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
# SIMPLE PYTHON SCRIPT TO PLOT .raw weight storage files. Input is path to
# weight files up to the specifier of whether they are weights or adam params.
# (eg /path/to/dir/agent_00_net_)

import sys
import numpy as np
import matplotlib.pyplot as plt
FILE=    sys.argv[1]
#ftype=np.float64
ftype=np.float32
W  = np.fromfile(FILE+"weights.raw", dtype=ftype)
M1 = np.fromfile(FILE+"diagCov.raw", dtype=ftype)
M2 = np.fromfile(FILE+"pathCov.raw", dtype=ftype)

plt.subplot(311)
plt.semilogy(np.abs(W),'k.')
plt.subplot(312)
plt.semilogy(M1, 'b.')
plt.subplot(313)
plt.plot(M2, 'r.')

plt.show()
