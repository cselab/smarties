//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Quadratic_term_h
#define smarties_Quadratic_term_h

#include "../Network/Layers/Functions.h"

namespace smarties
{

struct Quadratic_term
{
  using PosDefFunction = SoftPlus;
  const Uint start_matrix, start_mean, nA, nL;
  const Rvec netOutputs;
  const Rvec L, mean, matrix;
  static Uint compute_nL(const ActionInfo& aI)
  {
    return (aI.dim() * aI.dim() + aI.dim())/2;
  }

  Rvec getParam() const
  {
     Rvec ret(nL, 0);
     for (Uint ind=0, j=0; j<nA; ++j)
       for (Uint i=0; i<nA; ++i)
         if (i<j)       ret[ind++] = matrix[nA*j +i];
         else if (i==j) ret[ind++] = matrix[nA*j +i];
     return ret;
  }

  Quadratic_term(Uint _startMat, Uint _startMean, Uint _nA, Uint _nL,
    const Rvec& out, const Rvec _m = Rvec()) :
    start_matrix(_startMat), start_mean(_startMean), nA(_nA), nL(_nL),
    netOutputs(out), L(extract_L()), mean(extract_mean(_m)),
    matrix(extract_matrix())
    {
      //printf("Quadratic_term: %u %u %u %u %lu %lu %lu %lu\n", start_matrix,start_mean,nA,nL,
      //netOutputs.size(), L.size(), mean.size(), matrix.size());
      assert(L.size()==nA*nA && mean.size()==nA && matrix.size()==nA*nA);
      assert(netOutputs.size()>=start_matrix+nL && netOutputs.size()>=start_mean+nA);
    }

protected:
  Real quadMatMul(const Rvec& act, const Rvec& mat) const
  {
    assert(act.size() == nA && mat.size() == nA*nA);
    Real ret = 0;
    for (Uint j=0; j<nA; ++j)
    for (Uint i=0; i<nA; ++i)
      ret += (act[i]-mean[i])*mat[nA*j+i]*(act[j]-mean[j]);
    return ret;
  }

  Real quadraticTerm(const Rvec& act) const
  {
    return quadMatMul(act, matrix);
  }

  Rvec extract_L() const
  {
    assert(netOutputs.size()>=start_matrix+nL);
    Rvec ret(nA*nA);
    Uint kL = start_matrix;
    for (Uint j=0; j<nA; ++j)
    for (Uint i=0; i<nA; ++i)
      if (i<j)
        ret[nA*j + i] = netOutputs[kL++];
      else if (i==j)
        ret[nA*j + i] = PosDefFunction::_eval(netOutputs[kL++]);
    assert(kL==start_matrix+nL);
    return ret;
  }

  Rvec extract_mean(const Rvec tmp) const
  {
    //printf("%lu vec:%s\n", tmp.size(), print(tmp).c_str()); fflush(0);
    if(tmp.size() == nA) { assert(start_mean==0); return tmp; }
    assert(start_mean!=0 && netOutputs.size()>=start_mean+nA);
    return Rvec(&(netOutputs[start_mean]), &(netOutputs[start_mean])+nA);
  }

  Rvec extract_matrix() const //fill positive definite matrix P == L * L'
  {
    assert(L.size() == nA*nA);
    Rvec ret(nA*nA,0);
    for (Uint j=0; j<nA; ++j)
    for (Uint i=0; i<nA; ++i)
    for (Uint k=0; k<nA; ++k) {
      const Uint k1 = nA*j + k;
      const Uint k2 = nA*i + k;
      ret[nA*j + i] += L[k1] * L[k2];
    }
    return ret;
  }

  void grad_matrix(const Rvec&dErrdP, Rvec&netGradient) const
  {
    assert(netGradient.size() >= start_matrix+nL);
    for (Uint il=0; il<nL; il++)
    {
      Uint kL = 0;
      Rvec _dLdl(nA*nA, 0);
      for (Uint j=0; j<nA; ++j)
      for (Uint i=0; i<nA; ++i)
        if(i<=j) if(kL++==il) _dLdl[nA*j+i]=1;
      assert(kL==nL);

      netGradient[start_matrix+il] = 0;
      //_dPdl = dLdl' * L + L' * dLdl
      for (Uint j=0; j<nA; ++j)
      for (Uint i=0; i<nA; ++i)
      {
        Real dPijdl = 0;
        for (Uint k=0; k<nA; ++k)
        {
          const Uint k1 = nA*j + k;
          const Uint k2 = nA*i + k;
          dPijdl += _dLdl[k1]*L[k2] + L[k1]*_dLdl[k2];
        }
        netGradient[start_matrix+il] += dPijdl*dErrdP[nA*j+i];
      }
    }
    {
      Uint kl = start_matrix;
      for (Uint j=0; j<nA; ++j)
      for (Uint i=0; i<nA; ++i) {
        if (i==j) netGradient[kl] *= PosDefFunction::_evalDiff(netOutputs[kl]);
        if (i<j)  netGradient[kl] *= 1;
        if (i<=j) kl++;
      }
      assert(kl==start_matrix+nL);
    }
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
