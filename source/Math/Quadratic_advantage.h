//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Gaussian_policy.h"
#include "Quadratic_term.h"

struct Quadratic_advantage: public Quadratic_term
{
  const ActionInfo* const aInfo;
  const Gaussian_policy* const policy;

  //Normalized quadratic advantage, with own mean
  Quadratic_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
  const Rvec& out, const Gaussian_policy*const pol = nullptr) :
  Quadratic_term(starts[0],starts.size()>1? starts[1]:0, aI->dim,compute_nL(aI),
   out, pol==nullptr? Rvec(): pol->mean), aInfo(aI), policy(pol) {}

  inline void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const
  {
    assert(act.size()==nA);
    Rvec dErrdP(nA*nA, 0), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; j++) dAct[j] = act[j] - mean[j];

    assert(policy == nullptr);
    //for (Uint j=0; j<nA; j++) dPol[j] = policy->mean[j] - mean[j];

    for (Uint i=0; i<nA; i++)
    for (Uint j=0; j<=i; j++) {
      Real dOdPij = -dAct[j] * dAct[i];

      dErrdP[nA*j +i] = Qer*dOdPij;
      dErrdP[nA*i +j] = Qer*dOdPij; //if j==i overwrite, avoid `if'
    }

    for (Uint j=0, kl = start_matrix; j<nA; j++)
    for (Uint i=0; i<=j; i++) {
      Real dErrdL = 0;
      for (Uint k=i; k<nA; k++) dErrdL += dErrdP[nA*j +k] * L[nA*k +i];

      if(i==j) netGradient[kl] = dErrdL * unbPosMap_diff(netOutputs[kl]);
      else
      if(i<j)  netGradient[kl] = dErrdL;
      kl++;
    }

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (Uint a=0; a<nA; a++) {
        Real val = 0;
        for (Uint i=0; i<nA; i++)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[start_mean+a] = val;
        if(aInfo->bounded[a])
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

  inline Real computeAdvantage(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    if(policy not_eq nullptr)
    { //subtract expectation from advantage of action
      ret += quadraticTerm(policy->mean);
      for(Uint i=0; i<nA; i++)
        ret += matrix[nA*i+i] * policy->variance[i];
    }
    return 0.5*ret;
  }

  inline Real computeAdvantageNoncentral(const Rvec& action) const
  {
    Real ret = -quadraticTerm(action);
    return 0.5*ret;
  }

  inline Rvec getMean() const
  {
    return mean;
  }
  inline Rvec getMatrix() const
  {
    return matrix;
  }

  inline Real advantageVariance() const
  {
    if(policy == nullptr) return 0;
    Rvec PvarP(nA*nA, 0);
    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++)
    for (Uint k=0; k<nA; k++) {
      const Uint k1 = nA*j + k;
      const Uint k2 = nA*k + i;
      PvarP[nA*j+i] += matrix[k1] * policy->variance[k] * matrix[k2];
    }
    Real ret = quadMatMul(policy->mean, PvarP);
    for (Uint i=0; i<nA; i++)
      ret += 0.5 * PvarP[nA*i+i] * policy->variance[i];
    return ret;
  }

  void test(const Rvec& act, mt19937*const gen) const
  {
    Rvec _grad(netOutputs.size(), 0);
    grad(act, 1, _grad);
    ofstream fout("mathtest.log", ios::app);
    for(Uint i = 0; i<nL+nA; i++)
    {
      Rvec out_1 = netOutputs, out_2 = netOutputs;
      if(i>=nL && !start_mean) continue;
      const Uint index = i>=nL ? start_mean+i-nL : start_matrix+i;
      out_1[index] -= nnEPS;
      out_2[index] += nnEPS;

     Quadratic_advantage a1 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_1, policy);

     Quadratic_advantage a2 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_2, policy);

      const Real A_1 = a1.computeAdvantage(act);
      const Real A_2 = a2.computeAdvantage(act);
     {
       const double diffVal = (A_2-A_1)/(2*nnEPS);
       const double gradVal = _grad[index];
       const double errVal  = std::fabs(_grad[index]-(A_2-A_1)/(2*nnEPS));
       fout<<"Advantage grad "<<i<<" finite differences "
           <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
     }
    }
    fout.close();
  }
};
