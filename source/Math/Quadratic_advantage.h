//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Gaussian_policy.h"

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
    Rvec dErrdP(nA*nA), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; j++) dAct[j] = act[j] - mean[j];

    if(policy not_eq nullptr)
    for (Uint j=0; j<nA; j++) dPol[j] = policy->mean[j] - mean[j];

    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++) {
      const Real dOdPij = -.5*dAct[i]*dAct[j] + .5*dPol[i]*dPol[j]
        +.5*(i==j && policy not_eq nullptr ? policy->variance[i] : 0);

      dErrdP[nA*j+i] = Qer*dOdPij;
    }
    grad_matrix(dErrdP, netGradient);

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

  inline void grad_unb(const Rvec&act, const Real Qer,
    Rvec& netGradient) const
  {
    assert(act.size()==nA);
    Rvec dErrdP(nA*nA), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; j++) dAct[j] = act[j] - mean[j];

    if(policy not_eq nullptr)
    for (Uint j=0; j<nA; j++) dPol[j] = policy->mean[j] - mean[j];

    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++) {
      const Real dOdPij = -.5*dAct[i]*dAct[j] + .5*dPol[i]*dPol[j]
        +.5*(i==j && policy not_eq nullptr ? policy->variance[i] : 0);

      dErrdP[nA*j+i] = Qer*dOdPij;
    }
    grad_matrix(dErrdP, netGradient);

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (Uint a=0; a<nA; a++) {
        Real val = 0;
        for (Uint i=0; i<nA; i++)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[start_mean+a] = val;
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

  void test(const Rvec& act, mt19937*const gen) const;
};

struct Diagonal_advantage
{
  const ActionInfo* const aInfo;
  const Uint start_matrix, nA;
  const Rvec netOutputs;
  const Rvec mean, quadratic_coefs_pos, quadratic_coefs_neg;
  const Rvec linear_coefs_pos, linear_coefs_neg;
  const Gaussian_policy* const policy;
  static inline Uint compute_nL(const ActionInfo* const aI)
  {
    return 4*aI->dim;
  }

  Diagonal_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
    const Rvec& out, const Gaussian_policy*const pol) : aInfo(aI),
    start_matrix(starts[0]), nA(aI->dim), netOutputs(out), mean(pol->mean),
    quadratic_coefs_pos(extract(2)), quadratic_coefs_neg(extract(3)),
    linear_coefs_pos(extract(0)), linear_coefs_neg(extract(1)), policy(pol)
    { assert(starts.size()==1); }

 protected:
  inline Real diagMatMul(const Rvec& act) const
  {
    assert(act.size() == nA);
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Real u = act[i]-mean[i];
      if(u>0) ret -= u*u*quadratic_coefs_pos[i] + u*linear_coefs_pos[i];
      else    ret -= u*u*quadratic_coefs_neg[i] - u*linear_coefs_neg[i];
    }
    assert(ret<=0);
    return ret;
  }
  inline Rvec extract(const Uint i) const
  {
    const Uint start = start_matrix +i*nA;
    assert(netOutputs.size() >= start+nA);
    Rvec ret( &(netOutputs[start]), &(netOutputs[start+nA]) );
    for (Uint j=0; j<nA; j++) ret[j] = diag_func(ret[j]);
    return ret;
  }
  static inline Real diag_func(const Real val)
  {
    return 0.5*(val + std::sqrt(val*val+1));
  }
  static inline Real diag_func_diff(const Real val)
  {
    return 0.5*(1 + val/std::sqrt(val*val+1));
  }

 public:
  inline void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const
  {
    assert(act.size()==nA);
    for (Uint j=0; j<nA; j++)
    {
      const Real u = act[j] - mean[j], hvar = 0.5*policy->variance[j];
      if(u>0) {
        netGradient[start_matrix+0*nA+j] = -u;
        netGradient[start_matrix+2*nA+j] = -u*u;
        netGradient[start_matrix+1*nA+j] = netGradient[start_matrix+3*nA+j] = 0;
      } else {
        netGradient[start_matrix+0*nA+j] = netGradient[start_matrix+2*nA+j] = 0;
        netGradient[start_matrix+1*nA+j] = +u;
        netGradient[start_matrix+3*nA+j] = -u*u;
      }
      netGradient[start_matrix+0*nA+j] += std::sqrt(hvar/M_PI);
      netGradient[start_matrix+1*nA+j] += std::sqrt(hvar/M_PI);
      netGradient[start_matrix+2*nA+j] += hvar;
      netGradient[start_matrix+3*nA+j] += hvar;
    }

    for (Uint i=start_matrix; i<start_matrix + 4*nA; i++)
      netGradient[i] *= Qer*diag_func_diff(netOutputs[i]);
  }

  inline Real computeAdvantage(const Rvec& action) const
  {
    Real ret = diagMatMul(action);
    for (Uint i=0; i<nA; i++) { //add expectation from advantage of action
      const Real hvar = 0.5*policy->variance[i];
      ret += (quadratic_coefs_pos[i]+quadratic_coefs_neg[i])*hvar;
      ret += (linear_coefs_pos[i]+linear_coefs_neg[i])*std::sqrt(hvar/M_PI);
    }
    return ret;
  }

  inline Real advantageVariance() const
  {
    assert(policy not_eq nullptr);
    if(policy == nullptr) return 0;
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
    {
      const Real qp = quadratic_coefs_pos[i], lp = linear_coefs_pos[i];
      const Real qn = quadratic_coefs_neg[i], ln = linear_coefs_neg[i];
      const Real hvar = 0.5*policy->variance[i];
      const Real _EQ2 = 6*hvar*hvar*(qp*qp+qn*qn), _2EQ = pow(hvar*(qp+qn), 2);
      const Real _EL2 = hvar*(lp*lp+ln*ln), _2LQ = (hvar/M_PI)*pow(lp+ln, 2);
      ret += _EQ2-_2EQ + _EL2-_2LQ;
    }
    return ret;
  }
  void test(const Rvec& act, mt19937*const gen) const;
};
