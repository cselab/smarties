//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Continuous_policy_h
#define smarties_Continuous_policy_h

#include "../Network/Layers/Functions.h"
#include "../Core/Agent.h"

namespace smarties
{

template<typename T> T digamma(T x)
{
  T result = 0;
  assert(x > 0);
  for ( ; x < 7; ++x) result -= 1/x;
  x -= (T) 0.5;
  const T xx = 1/x, xx2 = xx*xx, xx4 = xx2*xx2;
  result += std::log(x) + (1.0/24.0)*xx2
        -(7.0/960.0)*xx4 +(31.0/8064.0)*xx4*xx2
        -(127.0/30720.0)*xx4*xx4;
  return result;
}

struct Base1Dpolicy
{
  const ActionInfo & aInfo;
  const size_t component_id, nnIndMean, nnIndStdev;

  virtual Real getMean() const = 0;
  virtual Real getStdev() const = 0;

  Base1Dpolicy(const ActionInfo & aI, const size_t comp,
                 const size_t startMean, const size_t startStdev) :
    aInfo(aI), component_id(comp), nnIndMean(startMean), nnIndStdev(startStdev)
  {
  }
  virtual ~Base1Dpolicy() {}

  virtual Real prob(const Rvec & act, const Rvec & beta_vec) const = 0;
  virtual Real prob(const Rvec & act) const  = 0;

  virtual Real logProb(const Rvec& act, const Rvec& beta_vec) const  = 0;
  virtual Real logProb(const Rvec& act) const  = 0;

  virtual Real KLdivergence(const Rvec& beta_vec) const  = 0;

  virtual std::array<Real,2> gradLogP(const Rvec& act, const Real factor) const = 0;
  virtual std::array<Real,2> gradKLdiv(const Rvec& act, const Real factor) const = 0;
  virtual std::array<Real,2> fixExplorationGrad(const Real targetNoise) const = 0;

  virtual Real sample(const Real noise) const = 0;
  virtual Real sample(std::mt19937& gen) const = 0;

  virtual Real sample_OrnsteinUhlenbeck(Rvec& state, const Real noise) const = 0;
  virtual Real sample_OrnsteinUhlenbeck(Rvec& state, std::mt19937& gen) const = 0;


  virtual void makeNetGrad(Rvec& nnGrad, const Rvec& nnOut, const Rvec& pGrad) const = 0;
};

struct NormalPolicy : public Base1Dpolicy
{
  using PosDefFunction = SoftPlus;
  const Real mean, stdev, invStdev = 1/stdev;

  Real getMean() const { return mean; }
  Real getStdev() const { return stdev; }
  Real linearNetToMean(const Rvec& nnOut) const {
    return nnOut[nnIndMean + component_id];
  }
  Real linearNetToStdev(const Rvec& nnOut) const {
    return PosDefFunction::_eval(nnOut[nnIndStdev + component_id]);
  }

  NormalPolicy(const ActionInfo & aI, const Rvec& nnOut, const size_t comp,
                const size_t startMean, const size_t startStdev) :
      Base1Dpolicy(aI, comp, startMean, startStdev),
      mean(linearNetToMean(nnOut)), stdev(linearNetToStdev(nnOut))
  {
  }

  static Real logProb(const Real a, const Real _mean, const Real _invStdev) {
    const Real arg = - std::pow( (a - _mean) * _invStdev, 2) / 2;
    //const Real fac = std::log(2 * M_PI) / 2; //log is not constexpr, equal:
    static constexpr Real fac = 9.1893853320467266954096885456237942e-01;
    return arg + std::log(_invStdev) - fac;
  }

  static Real prob(const Real a, const Real _mean, const Real _invStdev) {
    const Real arg = - std::pow( (a - _mean) * _invStdev, 2) / 2;
    //const Real fac = std::sqrt(1.0 / M_PI / 2); //sqrt is not constexpr, equal:
    static constexpr Real fac = 3.989422804014326857328237574407125976e-01;
    return _invStdev * fac * std::exp(arg);
  }

  Real prob(const Rvec & act, const Rvec & beta_vec) const {
    const Real beta_mean = beta_vec[component_id];
    const Real beta_stdev = beta_vec[component_id + aInfo.dim()];
    return prob(act[component_id], beta_mean, 1/beta_stdev);
  }

  Real prob(const Rvec & act) const {
    return prob(act[component_id], mean, invStdev);
  }

  Real logProb(const Rvec& act, const Rvec& beta_vec) const {
    const Real beta_mean = beta_vec[component_id];
    const Real beta_stdev = beta_vec[component_id + aInfo.dim()];
    return logProb(act[component_id], beta_mean, 1/beta_stdev);
  }

  Real logProb(const Rvec& act) const {
    return logProb(act[component_id], mean, invStdev);
  }

  Real KLdivergence(const Rvec& beta_vec) const {
    const Real beta_mean = beta_vec[component_id];
    const Real beta_stdev = beta_vec[component_id + aInfo.dim()];
    #ifndef SMARTIES_OPPOSITE_KL // do Dkl(mu||pi) :
      const Real CmuCpi = std::pow( beta_stdev * invStdev, 2);
      const Real sumDmeanC = std::pow((mean-beta_mean) * invStdev, 2);
    #else                        // do Dkl(pi||mu) :
      const Real CmuCpi = std::pow(stdev/beta_stdev, 2);
      const Real sumDmeanC = std::pow((mean-beta_mean)/beta_stdev, 2);
    #endif
    return ( CmuCpi-1 + sumDmeanC - std::log(CmuCpi) )/2;
  }

  std::array<Real, 2> gradLogP(const Rvec& act, const Real factor) const {
    const Real u = (act[component_id] - mean) * invStdev;
    const Real dLogPdMean = u * invStdev, dLogPdStdv = (u*u - 1) * invStdev;
    return {factor * dLogPdMean, factor * dLogPdStdv};
  }

  std::array<Real, 2> gradKLdiv(const Rvec& beta_vec, const Real factor) const {
    const Real beta_mean = beta_vec[component_id], dMean = mean - beta_mean;
    const Real beta_stdev = beta_vec[component_id + aInfo.dim()];
    #ifndef SMARTIES_OPPOSITE_KL // do grad Dkl(mu||pi) :
      const Real varMu = std::pow(beta_stdev, 2), var = stdev*stdev;
      const Real dDKLdMean = dMean * std::pow(invStdev, 2);
      const Real dDKLdStdv = (var - varMu - dMean*dMean) * std::pow(invStdev,3);
    #else                        // do grad Dkl(pi||mu) :
      const Real invVarMu = 1 / std::pow(beta_stdev, 2);
      const Real dDKLdMean = dMean * invVarMu;
      const Real dDKLdStdv = (invVarMu - std::pow(invStdev,2)) * stdev;
    #endif
    return {factor * dDKLdMean, factor * dDKLdStdv};
  }

  std::array<Real, 2> fixExplorationGrad(const Real targetNoise) const {
    return {0, (targetNoise - stdev) / 2};
  }

  static Real initial_Stdev(const ActionInfo& aI, Real explNoise) {
    return PosDefFunction::_inv(explNoise);
  }

  static Real sampleClippedGaussian(std::mt19937& gen) {
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);
    Real noise = dist(gen);
    if (noise >  NORMDIST_MAX || noise < -NORMDIST_MAX) return safety(gen);
    else return noise;
  }

  Real sample(const Real noise) const {
    return mean + stdev * noise;
  }

  Real sample(std::mt19937& gen) const {
    return sample(sampleClippedGaussian(gen));
  }

  Real sample_OrnsteinUhlenbeck(Rvec& state, const Real noise) const {
    const Real force = 0.85 * state[component_id];
    state[component_id] = noise + force; // update for next sample
    return mean + stdev * (noise + force);
  }

  Real sample_OrnsteinUhlenbeck(Rvec& state, std::mt19937& gen) const {
    const Real noise = sampleClippedGaussian(gen);
    return sample_OrnsteinUhlenbeck(state, noise);
  }

  void makeNetGrad(Rvec& nnGrad, const Rvec& nnOut, const Rvec& pGrad) const {
    assert(pGrad.size() == 2 * aInfo.dim());
    const auto indMean= nnIndMean+component_id, indStd= nnIndStdev+component_id;
    assert(nnOut.size() > indMean && nnGrad.size() > indMean);
    nnGrad[indMean] = pGrad[component_id];
    if(nnIndStdev == 0) return;
    assert(nnOut.size() > indStd && nnGrad.size() > indStd);
    const Real dPosdNet = PosDefFunction::_evalDiff(nnOut[indStd]);
    nnGrad[indStd] = dPosdNet * pGrad[component_id + aInfo.dim()];
  }
};

struct BetaPolicy : public Base1Dpolicy
{
  using ClipFunction = HardSigmoid;
  // Mean of beta distribution is in (0, 1)
  // Variance of beta distrubution is \in (0, mean*(1-mean))
  const Real mean, varCoef; // true variance = mean * (1-mean) * varCoef
  const Real stdev = std::sqrt(mean * (1-mean) * varCoef);
  const Real alpha =    mean  * (1/varCoef - 1);
  const Real beta  = (1-mean) * (1/varCoef - 1);

  // precomputed quantities:
  const Real M_DiGa  = digamma(alpha);
  const Real M_DiGb  = digamma(beta);
  const Real M_DiGab = digamma(alpha+beta);
  const Real M_logB = logB_func(alpha, beta);
  mutable bool storedOnBetaDigamma = false;
  mutable Real M_BetaA=0, M_BetaB=0, M_Beta_logB = 0;
  mutable Real M_BetaDiGa=0, M_BetaDiGb=0, M_BetaDiGab=0;

  Real getMean() const { return mean; }
  Real getStdev() const { return stdev; }

  Real linearNetToMean(const Rvec& nnOut) const {
    return ClipFunction::_eval(nnOut[nnIndMean + component_id]);
  }
  Real linearNetToVarCoef(const Rvec& nnOut) const {
    return ClipFunction::_eval(nnOut[nnIndStdev +component_id]);
  }

  BetaPolicy(const ActionInfo & aI, const Rvec& nnOut, const size_t comp,
              const size_t startMean, const size_t startStdev) :
      Base1Dpolicy(aI, comp, startMean, startStdev),
      mean(linearNetToMean(nnOut)), varCoef(linearNetToVarCoef(nnOut))
  {
  }

  static Real B_func(const Real _alpha, const Real _beta) {
    return std::tgamma(_alpha) * std::tgamma(_beta) / std::tgamma(_alpha+_beta);
  }
  static Real logB_func(const Real _alpha, const Real _beta) {
    return std::lgamma(_alpha) + std::lgamma(_beta) - std::lgamma(_alpha+_beta);
  }

  static Real prob(const Real u, const Real alpha, const Real beta, const Real norm) {
    assert(u>0 && u<1);
    return std::pow(u, alpha-1) * std::pow(1-u, beta-1) / norm;
  }
  static Real logProb(const Real u, const Real alpha, const Real beta, const Real norm) {
    assert(u>0 && u<1);
    return (alpha-1)*std::log(u) +(beta-1)*std::log(1-u) - norm;
  }

  // Converts {mean,stdev} used for storage and NN output into {alpha,beta}
  void betaVec2alphaBeta(const Rvec& beta_vec) const {
    if(storedOnBetaDigamma == false) {
      const Real beta_mean  = beta_vec[component_id];
      const Real beta_stdev = beta_vec[component_id + aInfo.dim()];
      const Real beta_varCoef = beta_stdev * beta_stdev / (beta_mean * (1-beta_mean));
      assert(beta_mean>0 && beta_mean<1);
      assert(beta_varCoef>0 && beta_varCoef<1);
      M_BetaA = beta_mean * (1/beta_varCoef - 1);
      M_BetaB  = (1-beta_mean) * (1/beta_varCoef - 1);
      M_Beta_logB = logB_func(M_BetaA, M_BetaB);
      M_BetaDiGa  = digamma(M_BetaA);
      M_BetaDiGb  = digamma(M_BetaB);
      M_BetaDiGab = digamma(M_BetaA+M_BetaB);
      storedOnBetaDigamma = true;
    }
  }

  Real prob(const Rvec & act, const Rvec & beta_vec) const {
    betaVec2alphaBeta(beta_vec);
    return prob(act[component_id], M_BetaA, M_BetaB, std::exp(M_Beta_logB));
  }
  Real prob(const Rvec & act) const {
    return prob(act[component_id], alpha, beta, std::exp(M_logB));
  }

  Real logProb(const Rvec& act, const Rvec& beta_vec) const {
    betaVec2alphaBeta(beta_vec);
    return logProb(act[component_id], M_BetaA, M_BetaB, M_Beta_logB);
  }
  Real logProb(const Rvec& act) const {
    return logProb(act[component_id], alpha, beta, M_logB);
  }

  Real KLdivergence(const Rvec & beta_vec) const {
    betaVec2alphaBeta(beta_vec);
    const Real term1 = M_logB - M_Beta_logB;
    const Real term2 = (M_BetaA - alpha) * M_BetaDiGa;
    const Real term3 = (M_BetaB - beta) * M_BetaDiGb;
    const Real term4 = (alpha - M_BetaA + beta - M_BetaB) * M_BetaDiGab;
    return term1 + term2 + term3 + term4;
  }

  std::array<Real, 2> gradLogP(const Rvec& act, const Real factor) const {
    const Real u = act[component_id];
    const Real dLogPdAlpha = M_DiGab + std::log(u) - M_DiGa;
    const Real dLogPdBeta = M_DiGab + std::log(1-u) - M_DiGb;
    const Real dAlphadMean = (1/varCoef-1);
    const Real dAlphadVarC = -mean/(varCoef*varCoef);
    const Real dBetadMean  = (1-1/varCoef);
    const Real dBetadVarC  = (mean-1)/(varCoef*varCoef);
    const Real dLogPdMean = dLogPdAlpha*dAlphadMean + dLogPdBeta*dBetadMean;
    const Real dLogPdVarC = dLogPdAlpha*dAlphadVarC + dLogPdBeta*dBetadVarC;
    return {factor * dLogPdMean, factor * dLogPdVarC};
  }

  std::array<Real, 2> gradKLdiv(const Rvec& mu_vec, const Real factor) const {
    betaVec2alphaBeta(mu_vec);
    const Real dDKLdAlpha = M_DiGa - M_DiGab - M_BetaDiGa + M_BetaDiGab;
    const Real dDKLdBeta = M_DiGb - M_DiGab - M_BetaDiGb + M_BetaDiGab;
    const Real dAlphadMean = (1/varCoef-1);
    const Real dAlphadVarC = -mean/(varCoef*varCoef);
    const Real dBetadMean  = (1-1/varCoef);
    const Real dBetadVarC  = (mean-1)/(varCoef*varCoef);
    const Real dDKLdMean = dDKLdAlpha*dAlphadMean + dDKLdBeta*dBetadMean;
    const Real dDKLdVarC = dDKLdAlpha*dAlphadVarC + dDKLdBeta*dBetadVarC;
    return {factor * dDKLdMean, factor * dDKLdVarC};
  }

  std::array<Real, 2> fixExplorationGrad(const Real targetNoise) const {
    const Real dLossdStdev = (targetNoise - stdev) / 2;
    const Real dStdevdVarC = std::sqrt(varCoef * mean*(1-mean)) / varCoef / 2;
    return { (Real) 0, dLossdStdev * dStdevdVarC};
  }

  static Real initial_Stdev(const ActionInfo& aI, Real explNoise) {
    static constexpr Real EPS = std::numeric_limits<float>::epsilon();
    if(explNoise > 1 - EPS) {
      printf("Exploration factor for Beta distribution (settings parameter "
             "--explNoise) must be less than 1.\n");
      explNoise  = 1 - EPS;
    }
    return ClipFunction::_inv(explNoise * explNoise / 4);
  }

  Real sample_OrnsteinUhlenbeck(Rvec& state, const Real noise) const {
    return sample(noise);
  }
  Real sample(const Real noise) const {
    printf("Shared noise is not supported by non-Gaussian distributions\n");
    fflush(0); abort();
    return noise;
  }

  static Real sampleBeta(std::mt19937& gen, const Real alpha, const Real beta) {
    std::gamma_distribution<Real> gamma_alpha(alpha, 1);
    std::gamma_distribution<Real> gamma_beta(beta, 1);
    const Real sampleAlpha = gamma_alpha(gen), sampleBeta = gamma_beta(gen);
    const Real sample = sampleAlpha/(sampleAlpha + sampleBeta);
    assert(sample > 0 && sample < 1);
    return sample;
  }

  Real sample(std::mt19937& gen) const {
    return sampleBeta(gen, alpha, beta);
  }

  Real sample_OrnsteinUhlenbeck(Rvec& state, std::mt19937& gen) const {
    const Real noise = sampleBeta(gen, alpha, beta);
    const Real force = 0.15 * (state[component_id] - noise);
    // if force>0 (i.e. OrnUhl state > noise) then move state closer to 0
    if(force>0) state[component_id] = (1-force) * state[component_id];
    // if force<0 (i.e. OrnUhl state < noise) then move state closer to 1
    else        state[component_id] = (1+force) * state[component_id] - force;
    return noise + force;
  }

  void makeNetGrad(Rvec& nnGrad, const Rvec& nnOut, const Rvec& pGrad) const {
    assert(pGrad.size() == 2*aInfo.dim());
    const auto indMean= nnIndMean+component_id, indStd= nnIndStdev+component_id;
    assert(nnOut.size() > indMean && nnGrad.size() > indMean);
    const Real dClipMdNet = ClipFunction::_evalDiff(nnOut[indMean]);
    nnGrad[indMean] = dClipMdNet * pGrad[component_id];
    if(nnIndStdev == 0) return;
    assert(nnOut.size() > indStd && nnGrad.size() > indStd);
    const Real dClipVdNet = ClipFunction::_evalDiff(nnOut[indStd]);
    nnGrad[indStd] = dClipVdNet * pGrad[component_id + aInfo.dim()];
  }
};

struct Continuous_policy
{
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  const ActionInfo & aInfo;
  const Uint startMean, startStdev, nA;
  const Rvec netOutputs;
  const std::vector<std::unique_ptr<Base1Dpolicy>> policiesVector;

  std::vector<std::unique_ptr<Base1Dpolicy>> make_policies() {
    std::vector<std::unique_ptr<Base1Dpolicy>> ret;
    for (Uint i=0; i<aInfo.dim(); ++i) {
      if(aInfo.isBounded(i))
        ret.emplace_back(std::make_unique<BetaPolicy>(
          aInfo, netOutputs, i, startMean, startStdev) );
      else
        ret.emplace_back(std::make_unique<NormalPolicy>(
          aInfo, netOutputs, i, startMean, startStdev) );
    }
    assert(ret.size() == aInfo.dim());
    return ret;
  }

  static Uint compute_nA(const ActionInfo& aI) {
    assert(aI.dim()); return aI.dim();
  }
  static Uint compute_nPol(const ActionInfo & aI) {
    return 2 * aI.dim();
  }
  static void setInitial_noStdev(const ActionInfo& aI, Rvec& initBias) {
    for(Uint e=0; e<aI.dim(); e++) initBias.push_back(0);
  }
  static void setInitial_Stdev(const ActionInfo& aI, Rvec&O, Real S) {
    if(S < std::numeric_limits<float>::epsilon()) {
      printf("Tried to initialize invalid pos-def mapping. Unless not training "
             "this should not be happening. Revise setting explNoise.\n");
      S = std::numeric_limits<float>::epsilon();
    }
    for(Uint i=0; i<aI.dim(); ++i)
        if(aI.isBounded(i)) O.push_back(BetaPolicy::initial_Stdev(aI, S));
        else                O.push_back(NormalPolicy::initial_Stdev(aI, S));
  }
  static Rvec initial_Stdev(const ActionInfo& aI, const Real S) {
    Rvec ret; ret.reserve(aI.dim());
    setInitial_Stdev(aI, ret, S);
    return ret;
  }

  static Rvec map2unbounded(const ActionInfo& aI, const Rvec & action) {
    Rvec ret = action;
    assert(action.size() == aI.dim());
    for(Uint i=0; i<aI.dim(); ++i)
        if(aI.isBounded(i)) ret[i] = BetaPolicy::ClipFunction::_inv(ret[i]);
    return ret;
  }

  ~Continuous_policy() = default;
  Continuous_policy(const Continuous_policy &p) = delete;
  // : aInfo(p.aInfo),
  //  startMean(p.startMean), startStdev(p.startStdev),
  //  nA(aInfo.dim()), netOutputs(p.netOutputs), policiesVector(make_policies())
  //{ }
  Continuous_policy& operator=(const Continuous_policy &p) = delete;
  Continuous_policy(Continuous_policy && p) = default;
  Continuous_policy& operator=(Continuous_policy && p) = delete;

  Continuous_policy(
    const std::vector<Uint>& inds, const ActionInfo& aI, const Rvec& nnOut) :
    aInfo(aI), startMean(inds[0]), startStdev(inds.size()>1? inds[1] : 0),
    nA(aI.dim()), netOutputs(nnOut), policiesVector(make_policies()) { }

  Real importanceWeight(const Rvec& act, const Rvec& beta) const {
    Real logW = 0;
    for (const auto & pol : policiesVector)
      logW += pol->logProb(act) - pol->logProb(act, beta);
    return std::exp( logW > 7 ? 7 : (logW < -7 ? -7 : logW) );
  }

  Real evalBehavior(const Rvec& act, const Rvec& beta) const {
    Real prob  = 1;
    assert(act.size() == nA);
    for (const auto & pol : policiesVector) {
      prob *= pol->prob(act, beta);
      if(prob < EPS) prob = EPS; // prevent NaN caused by underflow
    }
    return prob;
  }
  Real evalProbability(const Rvec& act) const {
    Real prob  = 1;
    assert(act.size() == nA);
    for (const auto & pol : policiesVector) {
      prob *= pol->prob(act);
      if(prob < EPS) prob = EPS; // prevent NaN caused by underflow
    }
    return prob;
  }
  Real evalLogProbability(const Rvec& act) const {
    Real prob  = 1;
    assert(act.size() == nA);
    for (const auto & pol : policiesVector) prob += pol->logProb(act);
    return prob;
  }

  template<typename T>
  Real KLDivergence(const T*const tgt_pol) const {
    const Rvec vecTarget = tgt_pol->getVector();
    return KLDivergence(vecTarget);
  }
  Real KLDivergence(const Rvec& beta) const {
    Real kldiv = 0;
    for (const auto & pol : policiesVector) kldiv += pol->KLdivergence(beta);
    return kldiv;
  }

  Rvec policyGradient(const Rvec& act, const Real coef) const {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; ++i) {
      const std::array<Real,2> PGi = policiesVector[i]->gradLogP(act, coef);
      ret[i] = PGi[0]; ret[i + nA] = PGi[1];
    }
    return ret;
  }

  template<typename T>
  Rvec KLDivGradient(const T*const tgt_pol, const Real coef = 1) const {
    const Rvec vecTarget = tgt_pol->getVector();
    return KLDivGradient(vecTarget, coef);
  }
  Rvec KLDivGradient(const Rvec& beta, const Real coef = 1) const {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; ++i) {
      const std::array<Real,2> PGi = policiesVector[i]->gradKLdiv(beta, coef);
      ret[i] = PGi[0]; ret[i + nA] = PGi[1];
    }
    return ret;
  }

  Rvec fixExplorationGrad(const Real target) const {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; ++i) {
      const auto PGi = policiesVector[i]->fixExplorationGrad(target);
      ret[i] = PGi[0]; ret[i + nA] = PGi[1];
    }
    return ret;
  }

  void makeNetworkGrad(Rvec& netGradient, const Rvec& totPolicyGrad) const {
    assert(netGradient.size()>=startMean+nA && totPolicyGrad.size() == 2*nA);
    for (const auto & pol : policiesVector)
      pol->makeNetGrad(netGradient, netOutputs, totPolicyGrad);
  }

  Rvec makeNetworkGrad(const Rvec& totPolicyGrad) const {
    Rvec netGradient = totPolicyGrad;
    assert(startMean == 0 && totPolicyGrad.size() == 2*nA);
    for (const auto & pol : policiesVector)
      pol->makeNetGrad(netGradient, netOutputs, totPolicyGrad);
    return netGradient;
  }

  Rvec getVector() const {
    Rvec beta(2*nA);
    for (Uint i=0; i<nA; ++i) {
      beta[i] = policiesVector[i]->getMean();
      beta[i + nA] = policiesVector[i]->getStdev();
    }
    return beta;
  }

  Rvec getMean() const {
    Rvec mean(nA);
    for (Uint i=0; i<nA; ++i) mean[i] = getMean(i);
    return mean;
  }
  Real getMean(const Uint comp_id) const {
    return policiesVector[comp_id]->getMean();
  }
  Rvec getVariance() const {
    Rvec var(nA);
    for (Uint i=0; i<nA; ++i) var[i] = getVariance(i);
    return var;
  }
  Real getStdev(const Uint comp_id) const {
    return policiesVector[comp_id]->getStdev();
  }
  Real getVariance(const Uint comp_id) const {
    return std::pow(policiesVector[comp_id]->getStdev(), 2);
  }

  Rvec selectAction(Agent& agent, const bool bTrain) const {
    //if (not bTrain || not agent.trackEpisode) return getMean();
    if (not bTrain) return getMean();
    // else sample:
    Rvec act(nA);
    if (agent.MDP.bAgentsShareNoise) {
        const Rvec noiseVec = agent.sampleActionNoise();
        for (Uint i=0; i<nA; ++i)
            act[i] = policiesVector[i]->sample(noiseVec[i]);
    } else
        for (Uint i=0; i<nA; ++i)
            act[i] = policiesVector[i]->sample(agent.generator);
    return act;
  }

  Rvec selectAction_OrnsteinUhlenbeck(Agent& agent, const bool bTrain, Rvec& state) const {
    //if (not bTrain || not agent.trackEpisode) return getMean();
    if (not bTrain) return getMean();
    // else sample:
    Rvec act(nA);
    if (agent.MDP.bAgentsShareNoise) {
      const Rvec noiseVec = agent.sampleActionNoise();
      for (Uint i=0; i<nA; ++i)
        act[i] = policiesVector[i]->sample_OrnsteinUhlenbeck(state, noiseVec[i]);
    } else
      for (Uint i=0; i<nA; ++i)
        act[i] = policiesVector[i]->sample_OrnsteinUhlenbeck(state, agent.generator);
    return act;
  }

  Rvec sample(std::mt19937& gen) const {
    Rvec action(nA);
    for (Uint i=0; i<nA; ++i) action[i] = policiesVector[i]->sample(gen);
    return action;
  }

  void test(const Rvec& act, const Rvec& beta) const;
};

void testContinuousPolicy(std::mt19937& gen, const ActionInfo & aI);

} // end namespace smarties
#endif // smarties_Continuous_policy_h
