//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Gaussian_advantage_h
#define smarties_Gaussian_advantage_h

#include "Continuous_policy.h"

namespace smarties
{

struct Gaussian_advantage
{
  using PosDefFunction = SoftPlus;
  static Uint compute_nL(const ActionInfo& aI) {
    return 1 + 2*aI.dim();
  }

  Rvec getParam() const {
    Rvec ret = matrix;
    ret.insert(ret.begin(), coef);
    return ret;
  }

  static void setInitial(const ActionInfo& aI, Rvec& initBias) {
    initBias.push_back(-1);
    for(Uint e=1; e<compute_nL(aI); e++) initBias.push_back(1);
  }

  const Uint start_coefs, nA, nL;
  const Rvec netOutputs;
  const Real coef;
  const Rvec matrix;
  const ActionInfo& aInfo;
  const Continuous_policy * const policy;

  //Normalized quadratic advantage, with own mean
  Gaussian_advantage(const std::vector<Uint>& starts, const ActionInfo& aI,
                     const Rvec& out, const Continuous_policy*const pol) :
    start_coefs(starts[0]), nA(aI.dim()), nL(compute_nL(aI)), netOutputs(out),
    coef(extract_coefs(netOutputs, starts[0])),
    matrix(extract_matrix(netOutputs, starts[0], aI.dim())),
    aInfo(aI), policy(pol) {}

private:

  static Rvec extract_matrix(const Rvec& net, const Uint start, const Uint nA)
  {
    Rvec ret = Rvec(2*nA);
    for(Uint i=0; i<2*nA; ++i)
      ret[i] = PosDefFunction::_eval(net[start +1 +i]);

    return ret;
  }

  static Real extract_coefs(const Rvec& net, const Uint start)
  {
    return PosDefFunction::_eval(net[start]);
  }

  void grad_matrix(Rvec& G, const Real err) const
  {
    G[start_coefs] *= err * PosDefFunction::_evalDiff(netOutputs[start_coefs]);
    for (Uint i=0, ind=start_coefs+1; i<2*nA; ++i, ++ind)
       G[ind] *= err * PosDefFunction::_evalDiff(netOutputs[ind]);
  }

public:

  Real computeAdvantage(const Rvec& act) const
  {
    const Real shape = - diagInvMul(act, matrix, policy->getMean()) / 2;
    const Real ratio = coefMixRatio(matrix, policy->getVariance());
    return coef * ( std::exp(shape) - ratio );
  }

  Real coefMixRatio(const Rvec&A, const Rvec&V) const
  {
    Real ret = 1;
    for (Uint i=0; i<nA; ++i)
      ret *= std::sqrt(A[i]/(A[i]+V[i]))/2 +std::sqrt(A[i+nA]/(A[i+nA]+V[i]))/2;
    return ret;
  }

  void grad(const Rvec&a, const Real Qer, Rvec& G) const
  {
    assert(a.size()==nA);

    const Real shape = - diagInvMul(a, matrix, policy->getMean()) / 2;
    const Real orig = std::exp(shape);
    const Real expect = - coefMixRatio(matrix, policy->getVariance());
    G[start_coefs] += orig + expect;

    for (Uint i=0, ind=start_coefs+1; i<nA; ++i, ++ind) {
      const Real m = policy->getMean(i), p1 = matrix[i], p2 = matrix[i+nA];
      G[ind]   = a[i]>m ? orig * coef * std::pow((a[i]-m)/p1, 2) / 2 : 0;
      G[ind+nA]= a[i]<m ? orig * coef * std::pow((a[i]-m)/p2, 2) / 2 : 0;
      const Real S = policy->getVariance(i);
      // inv of the pertinent coefMixRatio
      const Real F = 2 / (std::sqrt(p1/(p1+S)) + std::sqrt(p2/(p2+S)));
      //the derivatives of std::sqrt(A[i]/(A[i]+V[i])/2
      const Real diff1 = S / std::sqrt(p1 * std::pow(p1+S, 3)) / 4;
      const Real diff2 = S / std::sqrt(p2 * std::pow(p2+S, 3)) / 4;
      G[ind]    += F * expect*coef * diff1;
      G[ind+nA] += F * expect*coef * diff2;
    }

    grad_matrix(G, Qer);
  }

  void test(const Rvec& act, std::mt19937& gen) const;

  Real diagInvMul(const Rvec& act,
    const Rvec& mat, const Rvec& mean) const
  {
    assert(act.size()==nA); assert(mean.size()==nA); assert(mat.size()==2*nA);
    Real ret = 0;
    for (Uint i=0; i<nA; ++i) {
      const Uint matind = act[i]>mean[i] ? i : i+nA;
      ret += std::pow(act[i]-mean[i], 2)/mat[matind];
    }
    return ret;
  }
};

void testGaussianAdvantage(std::vector<Uint> polInds, std::vector<Uint> advInds,
  std::vector<Uint> netOuts, std::mt19937& gen, const ActionInfo & aI);

} // end namespace smarties
#endif // smarties_Gaussian_advantage_h
