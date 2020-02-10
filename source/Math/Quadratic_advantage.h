//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Quadratic_advantage_h
#define smarties_Quadratic_advantage_h

#include "Continuous_policy.h"
#include "Quadratic_term.h"

namespace smarties
{

struct Quadratic_advantage: public Quadratic_term
{
  const ActionInfo& aInfo;
  const Continuous_policy* const policy;

  //Normalized quadratic advantage, with own mean
  Quadratic_advantage(const std::vector<Uint>&starts,
                      const ActionInfo& aI,
                      const Rvec& out,
                      const Continuous_policy*const pol = nullptr) :
    Quadratic_term(starts[0], starts.size()>1? starts[1] : 0,
                   aI.dim(), compute_nL(aI), out,
                   pol==nullptr ? Rvec(): pol->getMean()),
                   aInfo(aI), policy(pol)
  {
  }

  void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const
  {
    assert(act.size()==nA);
    Rvec dErrdP(nA*nA, 0), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; ++j) dAct[j] = act[j] - mean[j];

    assert(policy == nullptr);
    //for (Uint j=0; j<nA; ++j) dPol[j] = policy->mean[j] - mean[j];

    for (Uint i=0; i<nA; ++i)
    for (Uint j=0; j<=i; ++j) {
      Real dOdPij = -dAct[j] * dAct[i];

      dErrdP[nA*j +i] = Qer*dOdPij;
      dErrdP[nA*i +j] = Qer*dOdPij; //if j==i overwrite, avoid `if'
    }

    for (Uint j=0, kl = start_matrix; j<nA; ++j)
    for (Uint i=0; i<=j; ++i) {
      Real dErrdL = 0;
      for (Uint k=i; k<nA; ++k) dErrdL += dErrdP[nA*j +k] * L[nA*k +i];

      if(i==j)
        netGradient[kl] = dErrdL * PosDefFunction::_evalDiff(netOutputs[kl]);
      else
      if(i<j)
        netGradient[kl] = dErrdL;
      kl++;
    }

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (Uint a=0; a<nA; a++) {
        Real val = 0;
        for (Uint i=0; i<nA; ++i)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[start_mean+a] = val;
        if ( aInfo.isBounded(a) )
        {
          if(mean[a]> BOUNDACT_MAX && netGradient[start_mean+a]>0)
            netGradient[start_mean+a] = 0;
          else
          if(mean[a]<-BOUNDACT_MAX && netGradient[start_mean+a]<0)
            netGradient[start_mean+a] = 0;
        }
      }
    }
  }

  Real computeAdvantage(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    if(policy not_eq nullptr)
    { //subtract expectation from advantage of action
      ret += quadraticTerm(policy->getMean());
      for(Uint i=0; i<nA; ++i)
        ret += matrix[nA*i+i] * policy->getVariance(i);
    }
    return 0.5*ret;
  }

  Real computeAdvantageNoncentral(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    return ret / 2;
  }

  Rvec getMean() const
  {
    return mean;
  }
  Rvec getMatrix() const
  {
    return matrix;
  }

  Real advantageVariance() const
  {
    if(policy == nullptr) return 0;
    Rvec PvarP(nA*nA, 0);
    for (Uint j=0; j<nA; ++j)
    for (Uint i=0; i<nA; ++i)
    for (Uint k=0; k<nA; ++k) {
      const Uint k1 = nA*j + k;
      const Uint k2 = nA*k + i;
      PvarP[nA*j+i]+= matrix[k1] * std::pow(policy->getStdev(i),2) * matrix[k2];
    }
    Real ret = quadMatMul(policy->getMean(), PvarP);
    for (Uint i=0; i<nA; ++i)
      ret += PvarP[nA*i+i] * std::pow(policy->getStdev(i), 2) / 2;
    return ret;
  }

  void test(const Rvec& act, std::mt19937& gen) const;
};

void testQuadraticAdvantage(std::vector<Uint> polInds, std::vector<Uint> advInds,
  std::vector<Uint> netOuts, std::mt19937& gen, const ActionInfo & aI);

} // end namespace smarties
#endif // smarties_Quadratic_advantage_h
