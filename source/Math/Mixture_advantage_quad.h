//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#if 0
#include "Gaussian_mixture_trunc.h"
#else
#include "Gaussian_mixture.h"
#endif

template<Uint nExperts>
struct Mixture_advantage
{
public:
  const Uint start_matrix, nA, nL;
  const Rvec netOutputs;
  const array<Rvec, nExperts> L, matrix;
  const ActionInfo* const aInfo;
  const Gaussian_mixture<nExperts>* const policy;
  //const array<array<Real,nExperts>, nExperts> overlap;

  Rvec getParam() const {
    Rvec ret(nL, 0);
    for(Uint ind=0, e=0; e<nExperts; e++)
      for (Uint j=0; j<nA; j++)
        for (Uint i=0; i<=j; i++)
          if (i<j)       ret[ind++] = matrix[e][nA*j +i];
          else if (i==j) ret[ind++] = matrix[e][nA*j +i];
    return ret;
  }
  static void setInitial(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<nExperts; e++)
      for (Uint j=0; j<aI->dim; j++)
        for (Uint i=0; i<=j; i++)
          if (i<j)       initBias.push_back( 0);
          else if (i==j) initBias.push_back(-1);
  }

  //Normalized quadratic advantage, with own mean
  Mixture_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
   const Rvec& out, const Gaussian_mixture<nExperts>*const pol) :
   start_matrix(starts[0]), nA(aI->dim), nL(compute_nL(aI)), netOutputs(out),
   L(extract_L()), matrix(extract_matrix()), aInfo(aI), policy(pol) { }

private:
    /*
    inline array<array<Real,nExperts>, nExperts> computeOverlap() const
    {
      array<array<Real,nExperts>, nExperts> ret;
      for (Uint e1=0; e1<nExperts; e1++)
      for (Uint e2=0; e2<=e1; e2++) {
        assert(overlapWeight(e1, e2) == overlapWeight(e2, e1));
        const Real W = policy->experts[e1]*policy->experts[e2];
        if(e1 == e2) ret[e1][e2] = W * selfOverlap(e1, e2);
        else ret[e1][e2] = ret[e2][e1] = W * overlapWeight(e1, e2);
      }
      return ret;
    }
    */

    static inline Real quadMatMul(const Uint nA, const Rvec& act,
      const Rvec& mat, const Rvec& mean)
    {
      assert(act.size()==nA);assert(mean.size()==nA);assert(mat.size()==nA*nA);
      Real ret = 0;
      for (Uint j=0; j<nA; j++) {
        ret += std::pow(act[j]-mean[j],2)*mat[nA*j+j];
        for (Uint i=0; i<j; i++) //exploit symmetry
          ret += 2*(act[i]-mean[i])*mat[nA*j+i]*(act[j]-mean[j]);
      }
      return ret;
    }

    inline array<Rvec,nExperts> extract_L() const
    {
      array<Rvec,nExperts> ret;
      Uint kL = start_matrix;
      for (Uint e=0; e<nExperts; e++) {
        ret[e] = Rvec(nA*nA, 0);
        for (Uint j=0; j<nA; j++)
        for (Uint i=0; i<=j; i++)
          if (i<j) ret[e][nA*j +i] = netOutputs[kL++];
          else if (i==j) ret[e][nA*j +i] = unbPosMap_func(netOutputs[kL++]);
      }
      assert(kL==start_matrix+nL);
      return ret;
    }

    inline array<Rvec,nExperts> extract_matrix() const
    {
      array<Rvec,nExperts> ret;
      for (Uint e=0; e<nExperts; e++) {
        ret[e] = Rvec(nA*nA, 0);
        for (Uint j=0; j<nA; j++)
          for (Uint i=0; i<=j; i++) //exploit symmetry
          {
            for (Uint k=0; k<=j; k++)
              ret[e][nA*j +i] += L[e][nA*j +k] * L[e][nA*i +k];

            ret[e][nA*i +j] = ret[e][nA*j +i]; //exploit symmetry
          }
      }
      return ret;
    }

public:
  inline void grad(const Rvec&act, const Real Qer, Rvec& netGradient) const
  {
    assert(act.size()==nA);
    for (Uint e=0, kl = start_matrix; e<nExperts; e++)
    {
      Rvec dErrdP(nA*nA, 0);

      for (Uint i=0; i<nA; i++)
      for (Uint j=0; j<=i; j++) {
        Real dOdPij = -policy->experts[e] *(act[j]-policy->means[e][j])
                                          *(act[i]-policy->means[e][i]);
        #if 1
        for (Uint ee=0; ee<nExperts; ee++) {
          const Real wght = policy->experts[e] * policy->experts[ee];
          dOdPij += wght * (policy->means[ee][j]-policy->means[e][j])
                         * (policy->means[ee][i]-policy->means[e][i]);
          if(i==j) dOdPij += wght * policy->variances[ee][i];
        }
        #endif

        dErrdP[nA*j +i] = Qer*dOdPij;
        dErrdP[nA*i +j] = Qer*dOdPij; //if j==i overwrite, avoid `if'
      }

      for (Uint j=0; j<nA; j++)
      for (Uint i=0; i<=j; i++) {
        Real dErrdL = 0;
        for (Uint k=i; k<nA; k++)
          dErrdL += dErrdP[nA*j +k] * L[e][nA*k +i];

        if(i==j) netGradient[kl] = dErrdL * unbPosMap_diff(netOutputs[kl]);
        else
        if(i<j)  netGradient[kl] = dErrdL;
        kl++;
      }
    }
  }
  inline Real computeAdvantage(const Rvec& action) const
  {
    Real ret = 0;
    for (Uint e=0; e<nExperts; e++) {
      ret-= policy->experts[e]*quadMatMul(nA,action,matrix[e],policy->means[e]);
      #if 1
      for (Uint ee=0; ee<nExperts; ee++) {
        const Real wght = policy->experts[e]*policy->experts[ee];
        for(Uint i=0; i<nA; i++)
         ret+= wght*matrix[e][nA*i+i] * policy->variances[ee][i];
        if(ee not_eq e)
         ret+= wght*quadMatMul(nA,policy->means[ee],matrix[e],policy->means[e]);
      }
      #endif
    }
    return 0.5*ret;
  }
  inline Real computeAdvantageNoncentral(const Rvec& action) const
  {
    Real ret = 0;
    for (Uint e=0; e<nExperts; e++)
    ret -= policy->experts[e]*quadMatMul(nA,action,matrix[e],policy->means[e]);
    return 0.5*ret;
  }

  static inline Uint compute_nL(const ActionInfo* const aI)
  {
    return nExperts*(aI->dim*aI->dim + aI->dim)/2;
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
      const Uint index = start_matrix+i;
      out_1[index] -= 0.0001; out_2[index] += 0.0001;

      Mixture_advantage a1(vector<Uint>{start_matrix}, aInfo, out_1, policy);
      Mixture_advantage a2(vector<Uint>{start_matrix}, aInfo, out_2, policy);

      const double A_1 = a1.computeAdvantage(act), A_2 = a2.computeAdvantage(act);
      const double fdiff =(A_2-A_1)/.0002, abserr = std::fabs(_grad[index]-fdiff);
      const double scale = std::max(std::fabs(fdiff), std::fabs(_grad[index]));
      //if((abserr>1e-4 || abserr/scale>1e-1) && policy->PactEachExp[ie]>nnEPS)
      {
        fout<<"Adv grad "<<i<<" finite differences "<<fdiff<<" analytic "
          <<_grad[index]<<" error "<<abserr<<" "<<abserr/scale<<endl;
      }
    }
    fout.close();
    #if 0
    Real expect = 0, expect2 = 0;
    for(Uint i = 0; i<1e6; i++) {
      const auto sample = policy->sample(gen);
      const auto advant = computeAdvantage(sample);
      expect += advant; expect2 += advant*advant;
    }
    const Real stdef = sqrt(expect2/1e6), meanv = expect/1e6;
    printf("Ratio of expectation: %f, mean %f\n", meanv/stdef, meanv);
    #endif
  }
  inline Real matrixDotVar(const Rvec& mat, const Rvec& S) const
  {
    Real ret = 0;
    for(Uint i=0; i<nA; i++) ret += mat[nA*i+i] * S[i];
    return ret;
  }
  inline Real overlapWeight(const Uint e1, const Uint e2) const
  {
    Real W = 0;
    for(Uint i=0; i<nA; i++)
      W += logP1DGauss(policy->means[e1][i], policy->means[e2][i],
                       policy->variances[e1][i] +policy->variances[e2][i]);
    return std::exp(W);
  }
  inline Real selfOverlap(const Uint e1, const Uint e2) const
  {
    Real W = 1;
    for(Uint i=0; i<nA; i++)
      W *= 2*M_PI*(policy->variances[e1][i]+policy->variances[e2][i]);
    return 1/std::sqrt(W);
  }
  static inline Real logP1DGauss(const Real act,const Real mean,const Real var)
  {
    return -0.5*(std::log(2*var*M_PI) +std::pow(act-mean,2)/var);
  }
  inline Rvec mix2mean(const Rvec& m1, const Rvec& S1, const Rvec& m2, const Rvec& S2) const
  {
    Rvec ret(nA, 0);
    for(Uint i=0; i<nA; i++) ret[i] =(S2[i]*m1[i] + S1[i]*m2[i])/(S1[i]+S2[i]);
    return ret;
  }
  inline Rvec mix2vars(const Rvec&S1,const Rvec&S2)const
  {
    Rvec ret(nA, 0);
    for(Uint i=0; i<nA; i++) ret[i] =(S2[i]*S1[i])/(S1[i]+S2[i]);
    return ret;
  }
};
