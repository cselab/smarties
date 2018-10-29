//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Gaussian_policy.h"

struct Quadratic_advantage
{
  static inline Uint compute_nL(const ActionInfo* const aI) {
    return 1 + 2*aI->dim;
  }

  Rvec getParam() const {
    Rvec ret = matrix;
    ret.insert(ret.begin(), coef);
    return ret;
  }

  static void setInitial(const ActionInfo* const aI, Rvec& initBias) {
    initBias.push_back(-1);
    for(Uint e=1; e<compute_nL(aI); e++) initBias.push_back(1);
  }

  const Uint start_coefs, nA, nL;
  const Rvec netOutputs;
  const Real coef;
  const Rvec matrix;
  const ActionInfo * const aInfo;
  const Gaussian_policy * const policy;

  //Normalized quadratic advantage, with own mean
  Quadratic_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
   const Rvec& out, const Gaussian_policy*const pol) :
   start_coefs(starts[0]), nA(aI->dim), nL(compute_nL(aI)), netOutputs(out),
   coef(extract_coefs(netOutputs,starts[0])),
   matrix(extract_matrix(netOutputs,starts[0], aI->dim)),
   aInfo(aI), policy(pol) {}

private:
  static inline Rvec extract_matrix(const Rvec net, const Uint start, const Uint nA) {
    Rvec ret = Rvec(2*nA);
    for(Uint i=0; i<2*nA; i++)
      ret[i] = unbPosMap_func(net[start +1 +i]);

    return ret;
  }
  static inline Real extract_coefs(const Rvec net, const Uint start)  {
    return unbPosMap_func(net[start]);
  }

  inline void grad_matrix(Rvec& G, const Real err) const {
    G[start_coefs] *= err * unbPosMap_diff(netOutputs[start_coefs]);
    for (Uint i=0, ind=start_coefs+1; i<2*nA; i++, ind++)
       G[ind] *= err * unbPosMap_diff(netOutputs[ind]);
  }

public:

  inline Real computeAdvantage(const Rvec& act) const {
    const Real shape = -.5 * diagInvMul(act, matrix, policy->mean);
    const Real ratio = coefMixRatio(matrix, policy->variance);
    return coef * ( std::exp(shape) - ratio );
  }

  inline Real coefMixRatio(const Rvec&A, const Rvec&V) const {
    Real ret = 1;
    for (Uint i=0; i<nA; i++)
      ret *= std::sqrt(A[i]/(A[i]+V[i]))/2 +std::sqrt(A[i+nA]/(A[i+nA]+V[i]))/2;
    return ret;
  }

  inline void grad(const Rvec&a, const Real Qer, Rvec& G) const
  {
    assert(a.size()==nA);

    const Real shape = -.5 * diagInvMul(a, matrix, policy->mean);
    const Real orig = std::exp(shape);
    const Real expect = - coefMixRatio(matrix, policy->variance);
    G[start_coefs] += orig + expect;

    for (Uint i=0, ind=start_coefs+1; i<nA; i++, ind++) {
      const Real m = policy->mean[i], p1 = matrix[i], p2 = matrix[i+nA];
      G[ind]   = a[i]>m ? orig*coef * std::pow((a[i]-m)/p1, 2)/2 : 0;
      G[ind+nA]= a[i]<m ? orig*coef * std::pow((a[i]-m)/p2, 2)/2 : 0;
      const Real S = policy->variance[i];
      // inv of the pertinent coefMixRatio
      const Real F = 2 / (std::sqrt(p1/(p1+S)) + std::sqrt(p2/(p2+S)));
      //the derivatives of std::sqrt(A[i]/(A[i]+V[i])/2
      const Real diff1 = 0.25* S/std::sqrt(p1 * std::pow(p1+S, 3));
      const Real diff2 = 0.25* S/std::sqrt(p2 * std::pow(p2+S, 3));
      G[ind]    += F * expect*coef * diff1;
      G[ind+nA] += F * expect*coef * diff2;
    }

    grad_matrix(G, Qer);
  }

  void test(const Rvec& act, mt19937*const gen) const
  {
    const Uint numNetOutputs = netOutputs.size();
    Rvec _grad(numNetOutputs, 0);
    grad(act, 1, _grad);
    ofstream fout("mathtest.log", ios::app);
    for(Uint i = 0; i<nL; i++)
    {
      Rvec out_1 = netOutputs, out_2 = netOutputs;
      const Uint index = start_coefs+i;
      out_1[index] -= 0.0001; out_2[index] += 0.0001;

      Quadratic_advantage a1(vector<Uint>{start_coefs}, aInfo, out_1, policy);
      Quadratic_advantage a2(vector<Uint>{start_coefs}, aInfo, out_2, policy);
      const Real A_1 = a1.computeAdvantage(act), A_2 = a2.computeAdvantage(act);
      const Real fdiff =(A_2-A_1)/.0002, abserr = std::fabs(_grad[index]-fdiff);
      const Real scale = std::max(std::fabs(fdiff), std::fabs(_grad[index]));
      //if(abserr>1e-7 && abserr/scale>1e-4)
      {
        fout<<"Adv grad "<<i<<" finite differences "<<fdiff<<" analytic "
          <<_grad[index]<<" error "<<abserr<<" "<<abserr/scale<<endl;
      }
    }
    fout.close();
  }

  inline Real diagInvMul(const Rvec& act,
    const Rvec& mat, const Rvec& mean) const {
    assert(act.size()==nA); assert(mean.size()==nA); assert(mat.size()==2*nA);
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Uint matind = act[i]>mean[i] ? i : i+nA;
      ret += std::pow(act[i]-mean[i],2)/mat[matind];
    }
    return ret;
  }
};
