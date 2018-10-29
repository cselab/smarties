//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Optimizer.h"
#include "saruprng.h"

struct AdaMax {
  const nnReal eta, B1, B2, lambda, fac;
  AdaMax(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta), B1(beta1), B2(beta2), lambda(_lambda), fac(_fac) {}

#ifdef NDEBUG
  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
#endif
  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, nnReal&M3, const nnReal W) const
  {
    const nnReal DW = grad * fac;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = std::max(B2 * M2, std::fabs(DW));
    #ifdef NESTEROV_ADAM
      const nnReal numer = B1*M1 + (1-B1)*DW;
    #else
      const nnReal numer = M1;
    #endif
    #ifdef NET_L1_PENAL
      const nnReal penal = -(W>0 ? lambda : -lambda);
    #else
      const nnReal penal = - W*lambda;
    #endif
    return eta * (numer/std::max(M2, nnEPS) + penal);
  }
};

template <class T>
struct Entropy : public T {
  Saru& gen;
  const nnReal eps;
  Entropy(nnReal _eta, nnReal beta1, nnReal beta2, nnReal bett1, nnReal bett2,
    nnReal _lambda, nnReal _fac, Saru& _gen) : T(_eta, beta1, beta2, bett1,
    bett2, _lambda, _fac, _gen), gen(_gen), eps(_eta*_lambda) {}

  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, const nnReal W) {
    return T::step(grad, M1, M2, W) + eps * gen.d_mean0_var1();
  }
};

struct Adam {
  const nnReal eta, B1, B2, lambda, fac;
  Adam(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2),
    lambda(_lambda), fac(_fac) {}

#ifdef NDEBUG
  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
#endif
  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, nnReal&M3, const nnReal W) const
  {
    #ifdef NET_L1_PENAL
      const nnReal penal = -(W>0 ? lambda : -lambda);
    #else
      const nnReal penal = - W*lambda;
    #endif
    const nnReal DW = fac * grad + penal;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = B2 * M2 + (1-B2) * DW*DW;
    #ifdef NESTEROV_ADAM // No significant effect
      const nnReal numer = B1*M1 + (1-B1)*DW;
    #else
      const nnReal numer = M1;
    #endif
    #ifdef AMSGRAD
      // No statistical improvement over NIPS implementation. However, without
      // decay factor for M3 performance worsens noticeably. Probably because,
      // unlike supervised learning, data distribution in RL changes over time
      // increasing the kurtosis of the incoming gradients. 1e-4 decay factor
      // is smallest that doesnt ruin returns on gym. 1e-3 (=B2) is meaningless.
      M3 = std::max((1.-1e-4)*M3, M2);
      const nnReal ret = eta * numer / ( nnEPS + std::sqrt(M3) );
    #else
      #ifdef SAFE_ADAM //numerical safety, assumes that 1-beta2 = (1-beta1)^2/10
        assert( std::fabs( (1-B2) - 0.1*std::pow(1-B1,2) ) < nnEPS );
        M2 = M2 < M1*M1/10 ? M1*M1/10 : M2;
      #endif
      const nnReal ret = eta * numer / ( nnEPS + std::sqrt(M2) );
    #endif

    assert(not std::isnan(ret) && not std::isinf(ret));
    return ret;
  }
};

void AdamOptimizer::prepare_update(const Rvec&L)
{
  gradSum->reduceThreadsGrad(grads);

  if (learn_size > 1) { //add up gradients across master ranks
    MPI(Iallreduce, MPI_IN_PLACE, gradSum->params, gradSum->nParams, MPI_NNVALUE_TYPE, MPI_SUM, mastersComm, &paramRequest);
  }
  nStep++;
}

void AdamOptimizer::apply_update()
{
  if(nStep == 0) die("nStep == 0");
  if(learn_size > 1) {
    if(paramRequest == MPI_REQUEST_NULL)
      die("I am in finalize without having started a reduction");
    MPI(Wait, &paramRequest, MPI_STATUS_IGNORE);
  }
  #ifndef __EntropySGD
    using Algorithm = Adam;
  #else
    using Algorithm = Entropy<Adam>;
  #endif
  //update is deterministic: can be handled independently by each node
  //communication overhead is probably greater than a parallelised sum

  const Real factor = 1./batchSize;
  nnReal* const paramAry = weights->params;
  assert(eta < 2e-3); //super upper bound for NN, srsly
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;

  #pragma omp parallel
  {
    const Uint thrID = static_cast<Uint>(omp_get_thread_num());
    Saru gen(nStep, thrID, generators[thrID]()); //needs 3 seeds
    Algorithm algo(_eta,beta_1,beta_2,beta_t_1,beta_t_2,lambda,factor,gen);
    nnReal* const M1 = _1stMom->params;
    nnReal* const M2 = _2ndMom->params;
    #ifdef AMSGRAD
      nnReal* const M3 = _2ndMax->params; // holds max of second moment
    #else
      nnReal* const M3 = _2ndMom->params; // unused
    #endif
    nnReal* const G  = gradSum->params;

  #pragma omp for simd schedule(static) aligned(paramAry,M1,M2,M3,G:VEC_WIDTH)
    for (Uint i=0; i<weights->nParams; i++)
    paramAry[i] += algo.step(G[i], M1[i], M2[i], M3[i], paramAry[i]);
  }

  gradSum->clear();
  // Needed by Adam optimization algorithm:
  beta_t_1 *= beta_1;
  if (beta_t_1<nnEPS) beta_t_1 = 0;
  beta_t_2 *= beta_2;
  if (beta_t_2<nnEPS) beta_t_2 = 0;

  // update frozen weights:
  if(tgtUpdateAlpha > 0 && tgt_weights not_eq nullptr) {
    if (cntUpdateDelay == 0) {
      // the targetDelay setting param can be either >1 or <1.
      // if >1 then it means "every how many steps copy weight to tgt weights"
      cntUpdateDelay = tgtUpdateAlpha;
      if(tgtUpdateAlpha>=1) tgt_weights->copy(weights);
      else { // else is the learning rate of an exponential averaging
        nnReal* const targetAry = tgt_weights->params;
        #pragma omp parallel for simd schedule(static) aligned(paramAry,targetAry:VEC_WIDTH)
        for(Uint j=0; j<weights->nParams; j++)
          targetAry[j] += tgtUpdateAlpha*(paramAry[j] - targetAry[j]);
      }
    }
    if(cntUpdateDelay>0) cntUpdateDelay--;
  }
}

void AdamOptimizer::save(const string fname, const bool backup)
{
  weights->save(fname+"_weights");
  if(tgt_weights not_eq nullptr) tgt_weights->save(fname+"_tgt_weights");
  _1stMom->save(fname+"_1stMom");
  _2ndMom->save(fname+"_2ndMom");
  #ifdef AMSGRAD
  _2ndMax->save(fname+"_2ndMax");
  #endif

  if(backup) {
    ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    weights->save(fname+"_"+ss.str()+"_weights");
    _1stMom->save(fname+"_"+ss.str()+"_1stMom" );
    _2ndMom->save(fname+"_"+ss.str()+"_2ndMom" );
    #ifdef AMSGRAD
    _2ndMax->save(fname+"_"+ss.str()+"_2ndMax" );
    #endif
  }
}
int AdamOptimizer::restart(const string fname)
{
  int ret = 0;
  ret = weights->restart(fname+"_weights");
  if(tgt_weights not_eq nullptr) {
    int missing_tgt = tgt_weights->restart(fname+"_tgt_weights");
    if (missing_tgt) tgt_weights->copy(weights);
  }
  _1stMom->restart(fname+"_1stMom");
  _2ndMom->restart(fname+"_2ndMom");
  #ifdef AMSGRAD
  _2ndMax->restart(fname+"_2ndMax");
  #endif
  return ret;
}

Optimizer::Optimizer(const Settings&S, const Parameters*const W,
  const Parameters*const WT, const vector<Parameters*>& samples) :
mastersComm(MPIComDup(S.mastersComm)), learn_size(S.learner_size),
pop_size(S.ESpopSize), nThreads(S.nThreads), weights(W), tgt_weights(WT),
sampled_weights(samples), eta_init(S.learnrate), batchSize(S.batchSize),
bAsync(S.bAsync), mpi_mutex(S.mpi_mutex), lambda(S.nnLambda),
epsAnneal(S.epsAnneal), tgtUpdateAlpha(S.targetDelay) {}

AdamOptimizer::AdamOptimizer(const Settings&S, const Parameters*const W,
  const Parameters*const WT, const vector<Parameters*>& samples,
  const vector<Parameters*>&G, const Real B1, const Real B2) :
  Optimizer(S,W,WT,samples), beta_1(B1), beta_2(B2), generators(S.generators),
  grads(G) { }
