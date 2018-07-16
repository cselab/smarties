//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Gaussian_policy.h"
template<Uint nExperts>
struct Gaussian_mixture
{
public:
  const ActionInfo* const aInfo;
  const Uint iExperts, iMeans, iPrecs, nA, nP;
  //const Real retraceTrickPow = 1. / std::cbrt(nA);
  //const Real P_trunc = (1-std::erf(NORMDIST_MAX/std::sqrt(2)))/(2*NORMDIST_MAX);
  //const Real retraceTrickPow = 1. / std::sqrt(nA);
  //const Real retraceTrickPow = 1. / nA;
  const Rvec netOutputs;
  const array<Real, nExperts> unnorm;
  const Real normalization;
  const array<Real, nExperts> experts;
  const array<Rvec, nExperts> means, stdevs, variances, precisions;
  //not kosher stuff, but it should work, relies on ordering of operations:

  Rvec sampAct;
  long double sampPonPolicy = -1, sampPBehavior = -1;
  Real sampImpWeight=0, sampKLdiv=0;
  array<long double, nExperts> PactEachExp;

  static inline Uint compute_nP(const ActionInfo* const aI) {
    return nExperts*(1 + 2*aI->dim);
  }
  static inline Uint compute_nA(const ActionInfo* const aI) {
    assert(aI->dim);
    return aI->dim;
  }
  inline Rvec map_action(const Rvec& sent) const {
    return aInfo->getInvScaled(sent);
  }

  Gaussian_mixture(const vector <Uint> starts, const ActionInfo*const aI,
    const Rvec&out) : aInfo(aI),
    iExperts(starts[0]), iMeans(starts[1]), iPrecs(starts[2]), nA(aI->dim),
    nP(compute_nP(aI)), netOutputs(out), unnorm(extract_unnorm()),
    normalization(compute_norm()), experts(extract_experts()),
    means(extract_mean()), stdevs(extract_stdev()),
    variances(extract_variance()), precisions(extract_precision())  {}
  /*
  Gaussian_mixture(const Gaussian_mixture<nExperts>& c) :
  aInfo(c.aInfo), iExperts(c.iExperts), iMeans(c.iMeans), iPrecs(c.iPrecs),
  nA(c.nA), nP(c.nP), netOutputs(c.netOutputs), unnorm(c.unnorm),
  normalization(c.normalization), experts(c.experts), means(c.means),
  variances(c.variances), precisions(c.precisions), stdevs(c.stdevs)
  { die("no copyconstructing"); }
  */
  Gaussian_mixture& operator= (const Gaussian_mixture<nExperts>& c) { die("no copying"); }


private:
  inline array<Real,nExperts> extract_unnorm() const {
    array<Real, nExperts> ret;
    if(nExperts == 1) {ret[0] = 1; return ret;}
    for(Uint i=0;i<nExperts;i++) ret[i]=unbPosMap_func(netOutputs[iExperts+i]);
    return ret;
  }
  inline Real compute_norm() const {
    Real ret = 0;
    if(nExperts == 1) {return 1;}
    for (Uint j=0; j<nExperts; j++) {
      ret += unnorm[j];
      assert(unnorm[j]>=0);
    }
    return ret + std::numeric_limits<Real>::epsilon();
  }
  inline array<Real,nExperts> extract_experts() const  {
    array<Real, nExperts> ret;
    if(nExperts == 1) {ret[0] = 1; return ret;}
    assert(normalization>0);
    for(Uint i=0;i<nExperts;i++) ret[i] = unnorm[i]/normalization;
    return ret;
  }
  inline array<Rvec,nExperts> extract_mean() const
  {
    array<Rvec,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iMeans + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = Rvec(&(netOutputs[start]), &(netOutputs[start+nA]));
    }
    return ret;
  }
  #ifdef EXTRACT_COVAR
  inline array<Rvec,nExperts> extract_stdev() const
  {
    array<Rvec,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iPrecs + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = Rvec(nA);
      for (Uint j=0; j<nA; j++)
        ret[i][j] = std::sqrt(noiseMap_func(netOutputs[start+j]));
    }
    return ret;
  }
  inline array<Rvec,nExperts> extract_variance() const
  {
    array<Rvec,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iPrecs + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = Rvec(nA);
      for(Uint j=0; j<nA; j++)
        ret[i][j] = noiseMap_func(netOutputs[start+j]);
    }
    return ret;
  }
  #else
  inline array<Rvec,nExperts> extract_stdev() const
  {
    array<Rvec,nExperts> ret;
    for(Uint i=0; i<nExperts; i++) {
      const Uint start = iPrecs + i*nA;
      assert(netOutputs.size() >= start + nA);
      ret[i] = Rvec(nA);
      for (Uint j=0; j<nA; j++) ret[i][j] = noiseMap_func(netOutputs[start+j]);
    }
    return ret;
  }
  inline array<Rvec,nExperts> extract_variance() const
  {
    array<Rvec,nExperts> ret = stdevs; //take sqrt of variance
    for(Uint i=0; i<nExperts; i++)
      for(Uint j=0; j<nA; j++)
        ret[i][j] = stdevs[i][j] * stdevs[i][j];
    return ret;
  }
  #endif
  inline array<Rvec,nExperts> extract_precision() const
  {
    array<Rvec,nExperts> ret = variances; //take inverse of precision
    for(Uint i=0; i<nExperts; i++)
      for(Uint j=0; j<nA; j++) ret[i][j] = 1/variances[i][j];
    return ret;
  }
  inline long double oneDnormal(const Real act, const Real mean, const Real prec) const
  {
    const long double arg = .5 * std::pow(act-mean,2) * prec;
    #if 0
      const auto Pgaus = std::sqrt(1./M_PI/2)*std::exp(-arg);
      const Real Punif = arg<.5*NORMDIST_MAX*NORMDIST_MAX? P_trunc : 0;
      return std::sqrt(prec)*(Pgaus + Punif);
    #else
      return std::sqrt(prec/M_PI/2)*std::exp(-arg);
    #endif
  }

public:
  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<nExperts*(1 + aI->dim); e++)
      initBias.push_back(0);
  }
  static void setInitial_Stdev(const ActionInfo* const aI, Rvec& initBias, const Real explNoise)
  {
    for(Uint e=0; e<nExperts*aI->dim; e++)
    #ifdef EXTRACT_COVAR
      initBias.push_back(noiseMap_inverse(explNoise*explNoise));
    #else
      initBias.push_back(noiseMap_inverse(explNoise));
    #endif
  }

  template <typename T>
  inline string print(const array<T,nExperts> vals)
  {
    std::ostringstream o;
    for (Uint i=0; i<nExperts-1; i++) o << vals[i] << " ";
    o << vals[nExperts-1];
    return o.str();
  }

  inline void prepare(const Rvec& unscal_act, const Rvec& beta)
  {
    sampAct = map_action(unscal_act);
    sampPonPolicy = 0; //numeric_limits<Real>::epsilon();
    for(Uint j=0; j<nExperts; j++) {
      PactEachExp[j] = 1;
      for(Uint i=0; i<nA; i++)
        PactEachExp[j] *= oneDnormal(sampAct[i], means[j][i], precisions[j][i]);
      sampPonPolicy += PactEachExp[j] * experts[j];
    }
    assert(sampPonPolicy>=0);
    sampPBehavior = evalBehavior(sampAct, beta);
    sampImpWeight = sampPonPolicy / sampPBehavior;
    sampKLdiv = kl_divergence(beta);
    if(sampPonPolicy<0){printf("observed %g\n",(Real)sampPonPolicy);fflush(0);}
  }
private:
  inline long double evalBehavior(const Rvec& act, const Rvec& beta) const {
    long double p = 0;
    const Uint NA = act.size();
    for(Uint j=0; j<nExperts; j++) {
      long double pi = 1;
      for(Uint i=0; i<NA; i++) {
        const Real stdi  = beta[i +j*NA +nExperts*(1+NA)]; //then stdevs
        pi *= oneDnormal(act[i], beta[i +j*NA +nExperts], 1/(stdi*stdi));
      }
      p += pi * beta[j]; //beta[j] contains expert
    }
    assert(p>0);
    return p;
  }
  inline Real logProbability(const Rvec& act) const {
    long double P = 0;
    for(Uint j=0; j<nExperts; j++) {
      long double pi  = 1;
      for(Uint i=0; i<nA; i++)
        pi *= oneDnormal(act[i], means[j][i], precisions[j][i]);
      P += pi * experts[j];
    }
    return std::log(P);
  }

public:
  // Sampling clipped between -NORMDIST_MAX and NORMDIST_MAX to reduce kurtosis
  // ensure that on-policy returns have finite probability of occurring.
  // Truncated normal is approximated by resampling from uniform  samples that
  // exceed the boundaries: the resulting PDF almost exactly truncared normal.
  inline Rvec sample(mt19937*const gen, const Rvec& beta) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);
    std::discrete_distribution<Uint> dE(&beta[0], &beta[nExperts]);

    const Uint experti = nExperts>1 ? dE(*gen) : 0;
    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      const Uint indM = i +experti*nA +nExperts; //after experts come the means
      const Uint indS = i +experti*nA +nExperts*(nA+1); //after means, stdev
      ret[i] = beta[indM] + beta[indS] * samp;
    }
    return ret;
  }
  inline Rvec sample(mt19937*const gen) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);
    std::discrete_distribution<Uint> dE(&(experts[0]), &(experts[nExperts]));

    const Uint experti = nExperts>1 ? dE(*gen) : 0;
    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = means[experti][i] + stdevs[experti][i] * samp;
    }
    return ret;
  }

  inline Rvec policy_grad(const Rvec& act, const Real factor) const
  {
    Rvec ret(nExperts +2*nA*nExperts, 0);
    // if sampPonPolicy == 0 then rho == 0 then we can skip this:
    // (also, we would get a nan)
    for(Uint j=0; j<nExperts && sampPonPolicy>0; j++) {
      const long double normExpert = factor * PactEachExp[j]/sampPonPolicy;
      assert(PactEachExp[j] > 0);
      for(Uint i=0; i<nExperts; i++)
        ret[i] += normExpert * ((i==j)-experts[j])/normalization;

      const long double fac = normExpert * experts[j];
      for (Uint i=0; i<nA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(1+nA)*nExperts;
        const Real u = act[i]-means[j][i];
        //const Real P=sqrt(.5*preci/M_PI)*safeExp(-pow(act[i]-meani,2)*preci);
        ret[indM] = fac*precisions[j][i]*u;
        #ifdef EXTRACT_COVAR
          ret[indS] = fac*(u*u*precisions[j][i]-1)/variances[j][i]/2;
        #else
          ret[indS] = fac*(u*u*precisions[j][i]-1)/stdevs[j][i];
        #endif
      }
    }
    return ret;
  }

  inline Rvec div_kl_grad(const Gaussian_mixture*const pol_hat, const Real fac = 1) const {
    const Rvec vecTarget = pol_hat->getVector();
    return div_kl_grad(vecTarget, fac);
  }
  inline Rvec div_kl_grad(const Rvec&beta, const Real fac=1) const
  {
    Rvec ret(nExperts +2*nA*nExperts, 0);
    for(Uint j=0; j<nExperts; j++) {
      Real DKLe = 0;
      for (Uint i=0; i<nA; i++) {
        const Uint indM = i+j*nA +nExperts, indS = i+j*nA +(nA+1)*nExperts;
        const Real preci = precisions[j][i], prech = 1/std::pow(beta[indS],2);
        ret[indM]= fac*experts[j]*(means[j][i]-beta[indM])*prech;
        #ifdef EXTRACT_COVAR
          ret[indS] = fac*experts[j]*(prech-preci)/2;
        #else
          ret[indS] = fac*experts[j]*(prech-preci)*stdevs[j][i];
        #endif
        const Real R = prech*variances[j][i];
        DKLe += R-1-std::log(R) +std::pow(means[j][i]-beta[indM],2)*prech;
      }
      const Real logRhoBeta = std::log(experts[j]/beta[j]);
      const Real tmp = fac*(.5*DKLe +1 +logRhoBeta)/normalization;
      for (Uint i=0; i<nExperts; i++) ret[i] += tmp*((i==j)-experts[j]);
    }
    return ret;
  }

  inline Real kl_divergence_exp(const Uint expi, const Rvec&beta) const
  {
    Real DKLe = 0; //numeric_limits<Real>::epsilon();
    for (Uint i=0; i<nA; i++) {
      const Real prech = 1/std::pow(beta[i+expi*nA +nExperts*(1+nA)], 2);
      const Real R =prech*variances[expi][i], meanh = beta[i+expi*nA +nExperts];
      DKLe += R-1-std::log(R) +std::pow(means[expi][i]-meanh,2)*prech;
    }
    assert(DKLe>=0);
    return 0.5*DKLe;
  }
  inline Real kl_divergence(const Gaussian_mixture*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  inline Real kl_divergence(const Rvec&beta) const
  {
    Real r = 0;
    for(Uint j=0; j<nExperts; j++)
    r += experts[j]*(std::log(experts[j]/beta[j])+kl_divergence_exp(j,beta));
    return r;
  }

  inline void finalize_grad(const Rvec&grad, Rvec&netGradient) const
  {
    assert(grad.size() == nP);
    for(Uint j=0; j<nExperts; j++) {
      {
        const Real diff = unbPosMap_diff(netOutputs[iExperts+j]);
        netGradient[iExperts+j] = grad[j] * diff;
      }
      for (Uint i=0; i<nA; i++) {
        netGradient[iMeans +i+j*nA] = grad[i+j*nA +nExperts];
        //if bounded actions pass through tanh!
        //helps against NaNs in converting from bounded to unbounded action space:
        if(aInfo->bounded[i]) {
          if(means[j][i]> BOUNDACT_MAX && netGradient[iMeans +i+j*nA]>0)
            netGradient[iMeans +i+j*nA] = 0;
          else
          if(means[j][i]<-BOUNDACT_MAX && netGradient[iMeans +i+j*nA]<0)
            netGradient[iMeans +i+j*nA] = 0;
        }

        const Real diff = noiseMap_diff(netOutputs[iPrecs +i+j*nA]);
        netGradient[iPrecs +i+j*nA] = grad[i+j*nA +(nA+1)*nExperts] * diff;
      }
    }
  }

  inline Rvec getBest() const
  {
    const Uint bestExp = std::distance(experts.begin(), std::max_element(experts.begin(),experts.end()));
    return means[bestExp];
  }
  inline Rvec finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  { //scale back to action space size:
    sampAct = bSample ? sample(gen, beta) : getBest();
    return aInfo->getScaled(sampAct);
  }

  inline Rvec getVector() const
  {
    Rvec ret(nExperts +2*nA*nExperts);
    for(Uint j=0; j<nExperts; j++) {
      ret[j] = experts[j];
      for (Uint i=0; i<nA; i++) {
        ret[i+j*nA +nExperts]        =  means[j][i];
        ret[i+j*nA +nExperts*(nA+1)] = stdevs[j][i];
      }
    }
    return ret;
  }

  void test(const Rvec& act, const Gaussian_mixture*const pol_hat) const
  {
    test(act, pol_hat->getVector());
  }

  void test(const Rvec& act, const Rvec beta) const
  {
    Rvec _grad(netOutputs.size());
    const Rvec div_klgrad = div_kl_grad(beta);
    const Rvec policygrad = policy_grad(act, 1);
    const Uint NEA = nExperts*(1+nA);
    ofstream fout("mathtest.log", ios::app);
    for(Uint i = 0; i<nP; i++)
    {
      Rvec out_1 = netOutputs, out_2 = netOutputs;
      const Uint ind = i<nExperts? iExperts+i :
        (i<NEA? iMeans +i-nExperts : iPrecs +i-NEA);
      const Uint ie = i<nExperts? i : (i<NEA? (i-nExperts)/nA : (i-NEA)/nA);
      assert(ie<nExperts);
      //if(PactEachExp[ie]<1e-12 && ind >= nExperts) continue;

      out_1[ind] -= nnEPS; out_2[ind] += nnEPS;
      Gaussian_mixture p1(vector<Uint>{iExperts, iMeans, iPrecs}, aInfo, out_1);
      Gaussian_mixture p2(vector<Uint>{iExperts, iMeans, iPrecs}, aInfo, out_2);
      const auto p_1=p1.logProbability(act), p_2=p2.logProbability(act);
      {
       finalize_grad(policygrad, _grad);
       const auto fdiff =(p_2-p_1)/nnEPS/2, abserr =std::fabs(_grad[ind]-fdiff);
       const auto scale = std::max(std::fabs(fdiff), std::fabs(_grad[ind]));
       //if((abserr>1e-7 && abserr/scale>1e-4) && PactEachExp[ie]>nnEPS)
       fout<<"Pol "<<i<<" fdiff "<<fdiff<<" grad "<<_grad[ind]<<" err "<<abserr
       <<" "<<abserr/scale<<" "<<sampPonPolicy<<" "<<PactEachExp[ie]<<endl;
      }

      const auto d_1=p1.kl_divergence(beta), d_2=p2.kl_divergence(beta);
      {
       finalize_grad(div_klgrad, _grad);
       const auto fdiff =(d_2-d_1)/nnEPS/2, abserr =std::fabs(_grad[ind]-fdiff);
       const auto scale = std::max(std::fabs(fdiff), std::fabs(_grad[ind]));
       fout<<"DKL "<<i<<" fdiff "<<fdiff<<" grad "<<_grad[ind]<<" err "<<abserr
       <<" "<<abserr/scale<<" "<<sampPonPolicy<<" "<<PactEachExp[ie]<<endl;
      }
    }
    fout.close();
  }
};
