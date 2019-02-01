//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Utils.h"

struct Gaussian_policy
{
  const ActionInfo* const aInfo;
  const Uint start_mean, start_prec, nA;
  //const Real P_trunc = (1-std::erf(NORMDIST_MAX/std::sqrt(2)))/(2*NORMDIST_MAX);
  const Rvec netOutputs;
  const Rvec mean, stdev, variance, precision;

  Rvec sampAct;
  Real sampImpWeight=0, sampKLdiv=0;

 private:
  long double sampPonPolicy=0, sampPBehavior=0;

 public:

  inline Rvec map_action(const Rvec& sent) const {
    return aInfo->getInvScaled(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI) {
    assert(aI->dim); return aI->dim;
  }

  Gaussian_policy(const vector <Uint>& start, const ActionInfo*const aI,
    const Rvec&out) : aInfo(aI), start_mean(start[0]),
    start_prec(start.size()>1 ? start[1] : 0), nA(aI->dim), netOutputs(out),
    mean(extract_mean()), stdev(extract_stdev()),
    variance(extract_variance()), precision(extract_precision()) {}

 private:
  inline Rvec extract_mean() const {
    assert(netOutputs.size() >= start_mean + nA);
    return Rvec(&(netOutputs[start_mean]),&(netOutputs[start_mean+nA]));
  }
  inline Rvec extract_precision() const {
    Rvec ret(nA);
    assert(variance.size() == nA);
    for (Uint j=0; j<nA; j++) ret[j] = 1/variance[j];
    return ret;
  }
  #ifdef EXTRACT_COVAR
    inline Rvec extract_stdev() const {
      Rvec ret(nA);
      assert(netOutputs.size() >= start_prec + nA);
      for(Uint i=0; i<nA; i++)
        ret[i] = std::sqrt( noiseMap_func(netOutputs[start_prec+i]) );
      return ret;
    }
  #else
    inline Rvec extract_stdev() const {
      Rvec ret(nA);
      assert(netOutputs.size() >= start_prec + nA);
      for(Uint i=0; i<nA; i++) ret[i] = noiseMap_func(netOutputs[start_prec+i]);
      return ret;
    }
  #endif

  inline Rvec extract_variance() const {
    Rvec ret(nA);
    assert(stdev.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = stdev[i]*stdev[i];
    return ret;
  }

  static inline long double oneDnormal(const Real A,const Real M,const Real P) {
    const long double arg = .5 * std::pow(A-M,2) * P;
    return std::sqrt(P/M_PI/2)*std::exp(-arg);
  }

 public:
  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<aI->dim; e++) initBias.push_back(0);
  }
  static void setInitial_Stdev(const ActionInfo*const aI, Rvec&O, const Real S)
  {
    #ifdef EXTRACT_COVAR
      for(Uint e=0; e<aI->dim; e++) O.push_back(noiseMap_inverse(S*S));
    #else
      for(Uint e=0; e<aI->dim; e++) O.push_back(noiseMap_inverse(S));
    #endif
  }

  inline void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampPonPolicy = evalLogProbability(sampAct);
    sampPBehavior = evalLogBehavior(sampAct, beta);
    const auto arg = sampPonPolicy - sampPBehavior;
    const auto clipArg = arg>7? 7 : (arg<-7? -7 : arg);
    sampImpWeight = std::exp( clipArg ) ;
    sampKLdiv = kl_divergence(beta);
  }

  inline long double evalBehavior(const Rvec& act, const Rvec& beta) const {
    long double pi  = 1;
    assert(act.size() == nA);
    for(Uint i=0; i<nA; i++) {
      assert(beta[nA+i]>0);
      pi *= oneDnormal(act[i], beta[i], 1/(beta[nA+i]*beta[nA+i]) );
    }
    return pi;
  }

  inline long double evalProbability(const Rvec& act) const {
    long double pi  = 1;
    for(Uint i=0; i<nA; i++) pi *= oneDnormal(act[i], mean[i], precision[i]);
    return pi;
  }

  inline Real evalLogBehavior(const Rvec& A, const Rvec& beta) const {
    Real p = 0;
    for(Uint i=0; i<nA; i++) {
      const Real M = beta[i], s = beta[nA+i];
      p -= std::pow( (A[i]-M) / s, 2 ) + std::log( 2*s*s*M_PI );
    }
    return 0.5 * p;
  }

  inline Real evalLogProbability(const Rvec& act) const {
    Real p = 0;
    for(Uint i=0; i<nA; i++) {
      p -= precision[i] * std::pow(act[i]-mean[i], 2);
      p += std::log(0.5*precision[i]/M_PI);
    }
    return 0.5 * p;
  }

  static inline Rvec sample(mt19937*const gen, const Rvec& beta)
  {
    assert(beta.size() / 2 > 0 && beta.size() % 2 == 0);
    Rvec ret(beta.size()/2);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<beta.size()/2; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      ret[i] = beta[i] + beta[beta.size()/2 + i]*samp;
    }
    return ret;
  }
  inline Rvec sample(mt19937*const gen) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      ret[i] = mean[i] + stdev[i]*samp;
    }
    return ret;
  }

  inline Rvec policy_grad(const Rvec& A, const Real F) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real U = (A[i]-mean[i]) * precision[i];
      ret[i] = F * U;
      #ifdef EXTRACT_COVAR
        ret[i+nA] = F * ( (A[i]-mean[i])*U - 1 ) * precision[i] / 2;
      #else
        ret[i+nA] = F * ( (A[i]-mean[i])*U - 1 ) / stdev[i];
      #endif
    }
    return ret;
  }

  inline Rvec div_kl_grad(const Gaussian_policy*const MU,const Real F=1) const {
    const Rvec vecTarget = MU->getVector();
    return div_kl_grad(vecTarget, F);
  }
  inline Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real preci = 1/std::pow(beta[nA+i], 2);
      ret[i]   = fac * (mean[i]-beta[i])*preci;
      #ifdef EXTRACT_COVAR
        ret[i+nA] = fac * (preci-precision[i]) /2;
      #else
        ret[i+nA] = fac * (preci-precision[i]) * stdev[i];
      #endif
    }
    return ret;
  }

  inline Real kl_divergence(const Gaussian_policy*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  inline Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Real prech = 1/std::pow(beta[nA+i],2);
      const Real R = variance[i]*prech;
      ret += R -1 -std::log(R) +std::pow(mean[i]-beta[i],2)*prech;
    }
    return 0.5*ret;
  }

  Rvec updateOrUhState(Rvec& state, const Rvec beta, const Real fac) {
    for (Uint i=0; i<nA; i++) {
      const Real noise = sampAct[i] - mean[i];
      state[i] *= fac;
      sampAct[i] += state[i];
      state[i] += noise;
    }
    return aInfo->getScaled(sampAct);
  }

  inline void finalize_grad(const Rvec grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; j++) {
      netGradient[start_mean+j] = grad[j];
      //if bounded actions pass through tanh!
      //helps against NaNs in converting from bounded to unbounded action space:
      if(aInfo->bounded[j])  {
        if(mean[j]> BOUNDACT_MAX && grad[j]>0) netGradient[start_mean+j] = 0;
        else
        if(mean[j]<-BOUNDACT_MAX && grad[j]<0) netGradient[start_mean+j] = 0;
      }
    }

    for (Uint j=0, iS=start_prec; j<nA && start_prec != 0; j++, iS++) {
      assert(netGradient.size()>=start_prec+nA);
      netGradient[iS] = grad[j+nA] * noiseMap_diff(netOutputs[iS]);
    }
  }

  inline Rvec finalize_grad(const Rvec grad) const {
    Rvec ret = grad;
    for (Uint j=0; j<nA; j++) if(aInfo->bounded[j]) {
      if(mean[j]> BOUNDACT_MAX && grad[j]>0) ret[j]=0;
      else
      if(mean[j]<-BOUNDACT_MAX && grad[j]<0) ret[j]=0;
    }

    if(start_prec != 0)
    for (Uint j=0, iS=start_prec; j<nA; j++, iS++)
      ret[j+nA] = grad[j+nA] * noiseMap_diff(netOutputs[iS]);
    return ret;
  }

  inline Rvec getMean() const {
    return mean;
  }
  inline Rvec getPrecision() const {
    return precision;
  }
  inline Rvec getStdev() const {
    return stdev;
  }
  inline Rvec getVariance() const {
    return variance;
  }
  inline Rvec getBest() const {
    return mean;
  }
  inline Rvec finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  { //scale back to action space size:
    sampAct = bSample ? sample(gen, beta) : mean;
    return aInfo->getScaled(sampAct);
  }

  inline Rvec getVector() const {
    Rvec ret = getMean();
    ret.insert(ret.end(), stdev.begin(), stdev.end());
    return ret;
  }

  void test(const Rvec& act, const Rvec& beta) const;
};
