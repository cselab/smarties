//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Quadratic_term.h"

struct Gaussian_policy
{
  const ActionInfo* const aInfo;
  const Uint start_mean, start_prec, nA;
  const Real P_trunc = (1-std::erf(NORMDIST_MAX/std::sqrt(2)))/(2*NORMDIST_MAX);
  const Rvec netOutputs;
  const Rvec mean, stdev, variance, precision;

  Rvec sampAct;
  long double sampPonPolicy=0, sampPBehavior=0;
  Real sampImpWeight=0, sampKLdiv=0;

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
  inline Rvec extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return Rvec(&(netOutputs[start_mean]),&(netOutputs[start_mean+nA]));
  }
  inline Rvec extract_precision() const
  {
    Rvec ret(nA);
    assert(variance.size() == nA);
    for (Uint j=0; j<nA; j++) ret[j] = 1/variance[j];
    return ret;
  }
  inline Rvec extract_variance() const
  {
    Rvec ret(nA);
    assert(stdev.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = stdev[i]*stdev[i];
    return ret;
  }
  inline Rvec extract_stdev() const
  {
    if(start_prec == 0) return Rvec (nA, ACER_CONST_STDEV);
    Rvec ret(nA);
    assert(netOutputs.size() >= start_prec + nA);
    for(Uint i=0; i<nA; i++) ret[i] = noiseMap_func(netOutputs[start_prec+i]);
    return ret;
  }
  static inline long double oneDnormal(const Real act, const Real _mean, const Real _prec) //const
  {
    const long double arg = .5 * std::pow(act-_mean,2) * _prec;
    #if 0
      const auto Pgaus = std::sqrt(1./M_PI/2)*std::exp(-arg);
      const Real Punif = arg<.5*NORMDIST_MAX*NORMDIST_MAX? P_trunc : 0;
      return std::sqrt(_prec)*(Pgaus + Punif);
    #else
      return std::sqrt(_prec/M_PI/2)*std::exp(-arg);
    #endif
  }

public:
  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<aI->dim; e++) initBias.push_back(0);
  }
  static void setInitial_Stdev(const ActionInfo* const aI, Rvec& initBias, const Real std0)
  {
    for(Uint e=0; e<aI->dim; e++) initBias.push_back(noiseMap_inverse(std0));
  }
  inline void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampPonPolicy = evalProbability(sampAct);
    sampPBehavior = evalBehavior(sampAct, beta);
    sampImpWeight = sampPonPolicy / sampPBehavior;
    sampKLdiv = kl_divergence(beta);
  }

  static inline double evalPolVec(const Rvec&act,const Rvec&mu,const Real stdev)
  {
    double pi  = 1, prec = 1/(stdev*stdev);
    for(Uint i=0; i<act.size(); i++) pi *= oneDnormal(act[i], mu[i], prec);
    return pi;
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

  inline Real logProbability(const Rvec& act) const {
    return std::log(evalProbability(act));
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
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
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
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = mean[i] + stdev[i]*samp;
    }
    return ret;
  }

  inline Rvec control_grad(const Quadratic_term*const adv, const Real eta) const
  {
    Rvec ret(nA*2, 0);
    for (Uint j=0; j<nA; j++) {
      for (Uint i=0; i<nA; i++)
        ret[j] += eta *adv->matrix[nA*j+i] *(adv->mean[i] - mean[i]);

      ret[j+nA] = .5*eta * adv->matrix[nA*j+j]*variance[j]*variance[j];
    }
    return ret;
  }

  inline Rvec policy_grad(const Rvec& act, const Real factor) const
  {
    /*
      this function returns factor * grad_phi log(policy(a,s))
      assumptions:
        - we deal with diagonal covariance matrices
        - network outputs the inverse of diag terms of the cov matrix
      Therefore log of distrib becomes:
      sum_i( -.5*log(2*M_PI*Sigma_i) -.5*(a-pi)^2*Sigma_i^-1 )
     */
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]    = factor*(act[i]-mean[i])*precision[i];
      ret[i+nA] = factor*(std::pow(act[i]-mean[i],2)*precision[i]-1)/stdev[i];
    }
    return ret;
  }

  inline Rvec div_kl_grad(const Gaussian_policy*const pol_hat, const Real fac = 1) const {
    const Rvec vecTarget = pol_hat->getVector();
    return div_kl_grad(vecTarget, fac);
  }
  inline Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real preci = 1/std::pow(beta[nA+i],2);
      ret[i]   = fac * (mean[i]-beta[i])*preci;
      ret[i+nA]= fac * (preci-precision[i])*stdev[i];
    }
    return ret;
  }
  static inline Rvec actDivKLgrad(const Rvec&pol, const Rvec&beta, const Real fac = 1)
  {
    assert(pol.size()*2 == beta.size());
    Rvec ret(pol.size());
    for (Uint i=0; i<pol.size(); i++)
      ret[i]   = fac * (pol[i]-beta[i])/std::pow(beta[pol.size()+i],2);
    return ret;
  }
  static inline Real actKLdivergence(const Rvec&pol, const Rvec& beta) {
    Real ret = 0;
    assert(pol.size()*2 == beta.size());
    for (Uint i=0; i<pol.size(); i++)
      ret += std::pow((pol[i]-beta[i])/beta[pol.size()+i],2);
    return 0.5*ret;
  }

  inline Real kl_divergence(const Gaussian_policy*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  inline Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Real prech = 1/std::pow(beta[nA+i],2), R = variance[i]*prech;
      ret += R -1 -std::log(R) +std::pow(mean[i]-beta[i],2)*prech;
    }
    return 0.5*ret;
  }

  Rvec updateOrUhState(Rvec& state, Rvec& beta, const Real fac)
  {
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
  inline Rvec finalize_grad(const Rvec&grad) const {
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
