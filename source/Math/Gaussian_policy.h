//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Gaussian_policy_h
#define smarties_Gaussian_policy_h

#include "../Network/Layers/Functions.h"

#ifndef PosDefMapping_f
#define PosDefMapping_f SoftPlus
#endif

namespace smarties
{

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

  Rvec map_action(const Rvec& sent) const
  {
    return aInfo->scaledAction2action(sent);
  }
  static Uint compute_nA(const ActionInfo* const aI)
  {
    assert(aI->dim()); return aI->dim();
  }

  static Real extract_stdev(const Real unbounbed)
  {
    #ifdef EXTRACT_COVAR
      return std::sqrt( PosDefMapping_f::_eval(unbounbed) );
    #else
      return            PosDefMapping_f::_eval(unbounbed)  ;
    #endif
  }

  Gaussian_policy(const std::vector<Uint>& start,
                    const ActionInfo*const aI,
                    const Rvec&out) :
    aInfo(aI), start_mean(start[0]),
    start_prec(start.size()>1 ? start[1] : 0),
    nA(aI->dim()), netOutputs(out),
    mean(extract_mean()), stdev(extract_stdev()),
    variance(extract_variance()),
    precision(extract_precision()) {}

 private:

  Rvec extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return Rvec(&(netOutputs[start_mean]),&(netOutputs[start_mean+nA]));
  }

  Rvec extract_precision() const
  {
    Rvec ret(nA);
    assert(variance.size() == nA);
    for (Uint j=0; j<nA; ++j) ret[j] = 1/variance[j];
    return ret;
  }

  Rvec extract_stdev() const
  {
    Rvec ret(nA);
    assert(netOutputs.size() >= start_prec + nA);
    for(Uint i=0; i<nA; ++i) ret[i] = extract_stdev(netOutputs[start_prec+i]);
    return ret;
  }

  Rvec extract_variance() const
  {
    Rvec ret(nA);
    assert(stdev.size() == nA);
    for(Uint i=0; i<nA; ++i) ret[i] = stdev[i]*stdev[i];
    return ret;
  }

  static long double oneDnormal(const Real A,const Real M,const Real P)
  {
    const long double arg = std::pow(A-M,2) * P / 2;
    return std::sqrt(P/M_PI/2)*std::exp(-arg);
  }

public:

  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<aI->dim(); e++) initBias.push_back(0);
  }

  static void setInitial_Stdev(const ActionInfo*const aI, Rvec&O, Real S)
  {
    if(S<=0) {
      printf("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.\n");
      S = std::numeric_limits<float>::epsilon();
    }
    #ifdef EXTRACT_COVAR
      const Real invFS = PosDefMapping_f::_inv(S*S);
    #else
      const Real invFS = PosDefMapping_f::_inv(S);
    #endif
    for(Uint e=0; e<aI->dim(); ++e) O.push_back(invFS);
  }

  static Rvec initial_Stdev(const ActionInfo*const aI, const Real S)
  {
    Rvec ret; ret.reserve(aI->dim());
    setInitial_Stdev(aI, ret, S);
    return ret;
  }

  void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampPonPolicy = evalLogProbability(sampAct);
    sampPBehavior = evalLogBehavior(sampAct, beta);
    const auto arg = sampPonPolicy - sampPBehavior;
    const auto clipArg = arg>7? 7 : (arg<-7? -7 : arg);
    sampImpWeight = std::exp( clipArg ) ;
    sampKLdiv = kl_divergence(beta);
  }

  long double evalBehavior(const Rvec& act, const Rvec& beta) const
  {
    long double pi  = 1;
    assert(act.size() == nA);
    for(Uint i=0; i<nA; ++i) {
      assert(beta[nA+i]>0);
      pi *= oneDnormal(act[i], beta[i], 1/(beta[nA+i]*beta[nA+i]) );
    }
    return pi;
  }

  long double evalProbability(const Rvec& act) const
  {
    long double pi  = 1;
    for(Uint i=0; i<nA; ++i) pi *= oneDnormal(act[i], mean[i], precision[i]);
    return pi;
  }

  Real evalLogBehavior(const Rvec& A, const Rvec& beta) const
  {
    Real p = 0;
    for(Uint i=0; i<nA; ++i) {
      const Real M = beta[i], s = beta[nA+i];
      p -= std::pow( (A[i]-M) / s, 2 ) + std::log( 2*s*s*M_PI );
    }
    return 0.5 * p;
  }

  Real evalLogProbability(const Rvec& act) const
  {
    Real p = 0;
    for(Uint i=0; i<nA; ++i) {
      p -= precision[i] * std::pow(act[i]-mean[i], 2);
      p += std::log(precision[i]/M_PI/2);
    }
    return p / 2;
  }

  static Rvec sample(std::mt19937*const gen, const Rvec& beta)
  {
    assert(beta.size() / 2 > 0 && beta.size() % 2 == 0);
    Rvec ret(beta.size()/2);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<beta.size()/2; ++i) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      ret[i] = beta[i] + beta[beta.size()/2 + i]*samp;
    }
    return ret;
  }
  Rvec sample(std::mt19937*const gen) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<nA; ++i) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      ret[i] = mean[i] + stdev[i]*samp;
    }
    return ret;
  }

  Rvec policy_grad(const Real F) const
  {
    return policy_grad(sampAct, F);
  }
  Rvec policy_grad(const Rvec& A, const Real F) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; ++i) {
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

  template<typename T>
  Rvec div_kl_grad(const T*const tgt_pol,const Real F=1) const
  {
    const Rvec vecTarget = tgt_pol->getVector();
    return div_kl_grad(vecTarget, F);
  }
  Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; ++i) {
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

  template<typename T>
  Real kl_divergence(const T*const tgt_pol) const
  {
    const Rvec vecTarget = tgt_pol->getVector();
    return kl_divergence(vecTarget);
  }
  Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; ++i) {
      const Real prech = 1/std::pow(beta[nA+i],2);
      const Real R = variance[i]*prech;
      ret += R -1 -std::log(R) +std::pow(mean[i]-beta[i],2)*prech;
    }
    return 0.5*ret;
  }

  Rvec updateOrUhState(Rvec& state, const Rvec beta, const Real fac)
  {
    for (Uint i=0; i<nA; ++i) {
      const Real noise = sampAct[i] - mean[i];
      state[i] *= fac;
      sampAct[i] += state[i];
      state[i] += noise;
    }
    return aInfo->action2scaledAction(sampAct);
  }

  void finalize_grad(const Rvec& grad, Rvec& netGradient) const
  {
    assert(netGradient.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; ++j) {
      netGradient[start_mean+j] = grad[j];
      //if bounded actions pass through tanh!
      //helps against NaNs in converting from bounded to unbounded action space:
      if( aInfo->isBounded(j) )  {
        if(mean[j]> BOUNDACT_MAX && grad[j]>0) netGradient[start_mean+j] = 0;
        else
        if(mean[j]<-BOUNDACT_MAX && grad[j]<0) netGradient[start_mean+j] = 0;
      }
    }

    for (Uint j=0, iS=start_prec; j<nA && start_prec != 0; ++j, iS++) {
      assert(netGradient.size()>=start_prec+nA);
      netGradient[iS] = grad[j+nA] * PosDefMapping_f::_evalDiff(netOutputs[iS]);
    }
  }

  Rvec finalize_grad(const Rvec& grad) const
  {
    Rvec ret = grad;
    for (Uint j=0; j<nA; ++j) if( aInfo->isBounded(j) ) {
      if(mean[j]> BOUNDACT_MAX && grad[j]>0) ret[j]=0;
      else
      if(mean[j]<-BOUNDACT_MAX && grad[j]<0) ret[j]=0;
    }

    if(start_prec != 0)
    for (Uint j=0, iS=start_prec; j<nA; ++j, iS++)
      ret[j+nA] = grad[j+nA] * PosDefMapping_f::_evalDiff(netOutputs[iS]);
    return ret;
  }

  Rvec getMean() const {
    return mean;
  }
  Rvec getPrecision() const {
    return precision;
  }
  Rvec getStdev() const {
    return stdev;
  }
  Rvec getVariance() const {
    return variance;
  }
  Rvec getBest() const {
    return mean;
  }
  Rvec finalize(const bool bSample, std::mt19937*const gen, Rvec& MU)
  { //scale back to action space size:
    for(Uint i=0; i<nA; ++i)
      if ( aInfo->isBounded(i) ) {
        MU[i] = std::max(-(Real)BOUNDACT_MAX, MU[i]);
        MU[i] = std::min( (Real)BOUNDACT_MAX, MU[i]);
      }
    sampAct = bSample ? sample(gen, MU) : mean;
    return aInfo->action2scaledAction(sampAct);
  }

  Rvec getVector() const {
    Rvec ret = getMean();
    ret.insert(ret.end(), stdev.begin(), stdev.end());
    return ret;
  }

  void test(const Rvec& act, const Rvec& beta) const;
};


} // end namespace smarties
#endif // smarties_Gaussian_policy_h
