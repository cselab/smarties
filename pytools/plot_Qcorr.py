#/usr/bin/env python
#
#  smarties
#  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
#  Distributed under the terms of the MIT license.
#
#  Created by Guido Novati (novatig@ethz.ch).
#
import numpy as np, matplotlib.pyplot as plt, sys
from scipy.stats import pearsonr

M    = int(sys.argv[1])
NFILES = len(range(2, len(sys.argv)))
means = np.zeros([M, NFILES])

for j in range(2, len(sys.argv)):
  DATA = np.fromfile(sys.argv[j]+'/onpolQdist.raw',dtype=np.float32)
  DATA = DATA.reshape([DATA.size // 4, 4])
  L = DATA.shape[0] // M
  DATA = DATA[DATA.shape[0] - M*L : -1, :]
  for i in range(M):
    start, end = i*L, (i+1)*L
    print(start, end)
    APOL = DATA[start:end, 0] #- DATA[start:end, 3]
    ANET = DATA[start:end, 1] + DATA[start:end, 3]
    ARET = DATA[start:end, 2] + DATA[start:end, 3]
    r = ((APOL - ARET) ** 2).mean()
    #r, p = pearsonr(APOL, ANET)
    means[i,j-2] = r

ret = np.mean(means, axis=1).reshape(M,1)
plt.semilogy(ret,'.')
plt.show()
