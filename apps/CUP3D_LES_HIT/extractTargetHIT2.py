#!/usr/bin/env python3.6
import re, argparse, numpy as np, glob

def findAllParams(path):
    NUs, EPSs, EXTs = set(), set(), set()
    alldirs = glob.glob(path+'*')
    for dirn in alldirs:
        EPSs.add(re.findall('EPS\d.\d\d\d',  dirn)[0][3:])
        NUs. add(re.findall('NU\d.\d\d\d\d', dirn)[0][2:])
    NUs  = list( NUs);  NUs.sort()
    EPSs = list(EPSs); EPSs.sort()
    return NUs, EPSs

def getAllFiles(dirsname, fSkip = 1):
    dirs = glob.glob(dirsname+'*')
    allFiles = []
    for path in dirs:
        files = glob.glob(path+'/analysis/spectralAnalysis_*')
        files.sort()
        allFiles = allFiles + files[::fSkip]
    return allFiles

def getAllData(files):
    nFiles = len(files)
    if nFiles == 0: return [], [], []

    def readData(fname, iscalars):
        f = open(fname, 'r')
        line = f.readline() # skip
        for nLine in range(15):
            line = f.readline()
            line = line.split()
            newKey = {line[0] : float(line[1])}
            iscalars.update(newKey)
        ks, en = np.loadtxt(fname, unpack=True, skiprows=18)
        return ks, en

    scalars = [dict() for i in range(nFiles)]
    modes, energy = readData(files[0], scalars[0])
    nModes = len(modes)
    spectra = np.ndarray(shape=(nFiles, nModes), dtype=float)
    spectra[0,:] = energy

    for i in range(1, nFiles):
        modes, energy = readData(files[i], scalars[i])
        spectra[i,:] = energy
    return modes, spectra, scalars

def computeIntTimeScale(scalars):
    tau_int = 0.0
    for i in range(len(scalars)):
      if(scalars[i]['tau_integral']<100): tau_int += scalars[i]['tau_integral']
      else: tau_int += 100000
    return tau_int/len(scalars)

def getLogSpectrumStats(spectrum):
    logE = np.log(np.asarray(spectrum))
    #print(logE.shape, spectrum.shape)
    mean  = np.array([np.mean(logE[:,i]) for i in range(spectrum.shape[1])])
    if False:
      stdev = np.array([np.std( logE[:,i]) for i in range(spectrum.shape[1])])
      for i in range(1, len(mean)):
        if mean[i] < -36: # exp(-36) < numerical precision
          mean[i]  = (mean[i] + mean[i-1])/2
          stdev[i] = max(stdev[i], stdev[i-1])
      #print(logE.shape, spectrum.shape, mean.shape, stdev.shape)
      return mean, stdev
    covar = np.cov(logE, rowvar=False)
    return mean, covar

def computeAverages(scalars):
    TK = [scalars[i]['tke']          for i in range(len(scalars))]
    LA = [scalars[i]['lambda']       for i in range(len(scalars))]
    RE = [scalars[i]['Re_lambda']    for i in range(len(scalars))]
    TI = [scalars[i]['tau_integral'] for i in range(len(scalars))]
    LI = [scalars[i]['l_integral']   for i in range(len(scalars))]
    GR = [scalars[i]['mean_grad']    for i in range(len(scalars))]
    EV = [scalars[i]['eps']          for i in range(len(scalars))]
    from numpy import mean, std
    return [mean(TK),mean(LA),mean(RE),mean(TI),mean(LI),mean(GR),mean(EV)], \
           [ std(TK), std(LA), std(RE), std(TI), std(LI), std(GR), std(EV)]

def main(path, fSkip):
  NUs, EPSs = findAllParams(path)

  for ei in range(len(EPSs)):
    EPSs[ei] = float(EPSs[ei])
    for ni in range(len(NUs)):
      NUs[ni] = float(NUs[ni])

      scalars, spectra, modes = [], None, None
      for run in [0, 1, 2, 3, 4]:
        dirn = '%sEXT2pi_EPS%.03f_NU%.04f_RUN%d' \
               % (path, EPSs[ei], NUs[ni], run)
        print(dirn)

        files = getAllFiles(dirn, fSkip = fSkip)
        runmodes, runspectra, runscalars = getAllData(files)
        if len(runmodes) == 0: continue

        tint = computeIntTimeScale(runscalars)
        tAnalysis = np.sqrt(NUs[ni] / EPSs[ei]) # time space between data files
        ind0 = int(5 * tint / tAnalysis / fSkip) # skip initial integral times

        scalars = scalars + runscalars[ind0:]
        modes = runmodes # modes are const in time, the box don't change
        if spectra is None: spectra = runspectra[ind0:]
        else: spectra = np.append(spectra, runspectra[ind0:], 0)

      if len(scalars) < 2: continue

      means, stdevs = computeAverages(scalars)
      modes *= 2 * np.pi / scalars[0]['lBox'] # from wave index to wave number
      avgLogSpec, covLogSpec = getLogSpectrumStats(spectra)

      fout = open('scalars_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), "w")
      fout.write( "tKinEn %e %e\n" % ( means[0], stdevs[0] ) )
      fout.write( "lambda %e %e\n" % ( means[1], stdevs[1] ) )
      fout.write( "Re_lam %e %e\n" % ( means[2], stdevs[2] ) )
      fout.write( "tInteg %e %e\n" % ( means[3], stdevs[3] ) )
      fout.write( "lInteg %e %e\n" % ( means[4], stdevs[4] ) )
      fout.write( "avg_Du %e %e\n" % ( means[5], stdevs[5] ) )
      fout.write( "epsVis %e %e\n" % ( means[6], stdevs[6] ) )

      nK = len(modes)
      print(modes.shape, avgLogSpec.shape)
      ary = np.append(modes.reshape([nK,1]), avgLogSpec.reshape([nK,1]), 1)
      np.savetxt('spectrumLogE_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), \
                 ary, delimiter=', ')
      np.savetxt('covarLogE_EPS%.03f_NU%.04f' % (EPSs[ei], NUs[ni]), \
                 covLogSpec, delimiter=', ')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description = "Compute a target file for RL agent from DNS data.")
  parser.add_argument('simdir',
      help="Simulation directory containing the 'Analysis' folder")
  parser.add_argument('--fSkip', type=int, default=1,
    help="Sampling frequency for analysis files. If 1, take all. " \
         "If 2, take 1 skip 1, If 3, take 1, skip 2, and so on.")
  args = parser.parse_args()

  main(args.simdir, args.fSkip)

