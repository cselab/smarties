//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MixedPG_h
#define smarties_MixedPG_h

#include "Learner_approximator.h"

namespace smarties
{

struct MixedPGstats
{
  Real Qerrm2 = 0;
  Rvec SPGm1, SPGm2, DPGm1, DPGm2;

  MixedPGstats(const Uint nA) : SPGm1(Rvec(nA, 0)),
    SPGm2(Rvec(nA, 0)), DPGm1(Rvec(nA, 0)), DPGm2(Rvec(nA, 0)) {}

  void add(const Rvec& SPG, const Rvec& DPG, const Real errQ) {
    Qerrm2   += std::pow(errQ, 2);
    for (Uint i = 0; i < DPG.size(); ++i) {
      SPGm1[i] += SPG[i];
      SPGm2[i] += SPG[i] * SPG[i];
      DPGm1[i] += DPG[i];
      DPGm2[i] += DPG[i] * DPG[i];
    }
  }

  static void update(Rvec & DPGfactor, Real & errQfactor,
                     std::vector<MixedPGstats> & stats, const Uint nA,
                     const Real learnRate, const Real batchSize)
  {
    Real varErrQ = 0;
    for (Uint j=0; j<stats.size(); ++j) {
      varErrQ += stats[j].Qerrm2 / batchSize; stats[j].Qerrm2 = 0;
    }
    errQfactor += learnRate * (varErrQ - errQfactor);

    for (Uint i = 0; i < nA; ++i) {
      Real meanDPG = 0, varDPG = 0, meanSPG = 0, varSPG = 0;
      for (Uint j = 0; j < stats.size(); ++j) {
        meanDPG += stats[j].DPGm1[i] / batchSize; stats[j].DPGm1[i] = 0;
        varDPG  += stats[j].DPGm2[i] / batchSize; stats[j].DPGm2[i] = 0;
        meanSPG += stats[j].SPGm1[i] / batchSize; stats[j].SPGm1[i] = 0;
        varSPG  += stats[j].SPGm2[i] / batchSize; stats[j].SPGm2[i] = 0;
      }
      //const Real stdDPG = std::sqrt(varDPG - meanDPG * meanDPG);
      const Real stdSPG = std::sqrt(varSPG - meanSPG * meanSPG);
      const Real newNorm = 0.2 * stdSPG / std::sqrt(varDPG + nnEPS);
      //const Real newNorm = std::sqrt(varSPG / (varDPG + nnEPS));
      DPGfactor[i] += learnRate * (newNorm - DPGfactor[i]);
    }
  }
};

class MixedPG : public Learner_approximator
{
  const Uint nA = aInfo.dim();
  const Real explNoise = settings.explNoise;
  Rvec DPGfactor = Rvec(nA, 0);
  Real errQfactor = 0;
  mutable std::vector<MixedPGstats> stats = std::vector<MixedPGstats>(nThreads, nA);

  Approximator* actor;
  Approximator* critc;

  void Train(const MiniBatch& MB, const Uint wID, const Uint bID) const override;

public:
  MixedPG(MDPdescriptor&, HyperParameters&, ExecutionInfo&);

  void setupTasks(TaskQueue& tasks) override;
  void selectAction(const MiniBatch& MB, Agent& agent) override;
  void processTerminal(const MiniBatch& MB, Agent& agent) override;
};

}

#endif // smarties_DPG_h
