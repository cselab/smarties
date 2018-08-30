//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"
#include <dirent.h>
#include <iterator>
#include <parallel/algorithm>

MemoryBuffer::MemoryBuffer(Environment* const _env, Settings & _s):
 mastersComm(_s.mastersComm), env(_env), bWriteToFile(_s.samplesFile),
 bTrain(_s.bTrain), bSampleSeq(_s.bSampleSequences), nAppended(_s.appendedObs),
 batchSize(_s.batchSize), maxTotObsNum(_s.maxTotObsNum), nThreads(_s.nThreads),
 policyVecDim(_s.policyVecDim), generators(_s.generators), gamma(_s.gamma),
 learn_rank(_s.learner_rank), learn_size(_s.learner_size){
  assert(_s.nAgents>0);
  inProgress.resize(_s.nAgents);
  for (int i=0; i<_s.nAgents; i++) inProgress[i] = new Sequence();
  Set.reserve(maxTotObsNum);
}

// Once learner receives a new observation, first this function is called
// to add the state and reward to the memory buffer
// this is called first also bcz memory buffer is used by net to pick new action
void MemoryBuffer::add_state(const Agent&a)
{
  if(a.Status < TERM_COMM) {
    nSeenTransitions ++;
  }

  #if 1
    if (inProgress[a.ID]->tuples.size() && a.Status == INIT_COMM) {
      //prev sequence not empty, yet received an initial state, push back prev
      warn("Unexpected termination of sequence");
      push_back(a.ID);
    } else if(inProgress[a.ID]->tuples.size()==0) {
      if(a.Status not_eq INIT_COMM) die("Missing initial state");
    }
  #endif

  #ifndef NDEBUG // check that last new state and new old state are the same
    if(inProgress[a.ID]->tuples.size()) {
      bool same = true;
      const Rvec vecSold = a.sOld.copy_observed();
      const auto memSold = inProgress[a.ID]->tuples.back()->s;
      for (Uint i=0; i<sI.dimUsed && same; i++) //scaled vec only has used dims:
        same = same && std::fabs(memSold[i]-vecSold[i]) < 1e-4;
      if (!same) { //create new sequence
        warn("Unexpected termination of sequence");
        push_back(a.ID);
      }
    }
  #endif

  // environment interface can overwrite reward. why? it can be useful.
  env->pickReward(a);
  inProgress[a.ID]->ended = a.Status==TERM_COMM;
  inProgress[a.ID]->add_state(a.s.copy_observed(), a.r);
}

// Once network picked next action, call this method
void MemoryBuffer::add_action(const Agent& a, const Rvec pol) const
{
  if(pol.size() not_eq policyVecDim) die("add_action");
  inProgress[a.ID]->add_action(a.a.vals, pol);
  if(bWriteToFile) a.writeData(learn_rank, pol, nSeenTransitions);
}

// If the state is terminal, instead of calling `add_action`, call this:
void MemoryBuffer::terminate_seq(Agent&a)
{
  assert(a.Status>=TERM_COMM);
  assert(inProgress[a.ID]->tuples.back()->mu.size() == 0);
  assert(inProgress[a.ID]->tuples.back()->a.size()  == 0);
  // fill empty action and empty policy:
  a.act(Rvec(aI.dim,0));
  inProgress[a.ID]->add_action(a.a.vals, Rvec(policyVecDim, 0));
  if(bWriteToFile)
    a.writeData(learn_rank, Rvec(policyVecDim, 0), nSeenTransitions);
  push_back(a.ID);
}

// update the second order moment of the rewards in the memory buffer
void MemoryBuffer::updateRewardsStats(unsigned long nStep, Real WR, Real WS)
{
  if(!bTrain) return; //if not training, keep the stored values
  if(WR<=0 && WS<=0) {
    debugL("Learner did not request any update to the state or rew stats");
    return;
  }
  WR = std::min((Real)1, WR);
  WS = std::min((Real)1, WS);
  long double count = 0, newstdvr = 0;
  vector<long double> newSSum(dimS, 0), newSSqSum(dimS, 0);
  #pragma omp parallel reduction(+ : count, newstdvr)
  {
    vector<long double> thNewSSum(dimS, 0), thNewSSqSum(dimS, 0);
    #pragma omp for schedule(dynamic)
    for(Uint i=0; i<Set.size(); i++) {
      count += Set[i]->ndata();
      for(Uint j=0; j<Set[i]->ndata(); j++) {
        newstdvr += std::pow(Set[i]->tuples[j+1]->r, 2);
        for(Uint k=0; k<dimS && WS>0; k++) {
          const long double sk = Set[i]->tuples[j]->s[k] - mean[k];
          thNewSSum[k] += sk; thNewSSqSum[k] += sk*sk;
        }
      }
    }
    if(WS>0) {
      #pragma omp critical
      for(Uint k=0; k<dimS; k++) {
        newSSum[k]   += thNewSSum[k];
        newSSqSum[k] += thNewSSqSum[k];
      }
    }
  }

  //add up gradients across nodes (masters)
  if (learn_size > 1) {
    LDvec res = LDvec(nReduce);
    res[0] = count; res[1] = newstdvr;
    for(Uint k=0; k<dimS; k++) {
      res[2+k] = newSSum[k]; res[2+dimS+k] = newSSqSum[k];
    }
    bool skipped = reductor.sync(res, WS>=1);
    if(skipped) {
      debugL("Update of state/reward data has been skipped");
      // typically off pol learner does an accurate reduction on first step
      // to compute state/rew statistics, then we can assume they change slowly
      // and if we have multiple ranks we use the result from previous reduction
      // in order to avoid blocking communication
      return; // no reduction done.
    }
    count = res[0]; newstdvr = res[1];
    for(Uint k=0; k<dimS; k++) {
      newSSum[k] = res[2+k]; newSSqSum[k] = res[2+dimS+k];
    }
  }

  if(count<batchSize) die("");
  static constexpr Real EPS = numeric_limits<float>::epsilon();
  if(WR>0)
  {
    long double varR = newstdvr/count;
    if(varR < numeric_limits<long double>::epsilon()) varR = 1;
    invstd_reward = (1-WR)*invstd_reward +WR/(std::sqrt(varR)+EPS);
  }
  for(Uint k=0; k<dimS && WS>0; k++)
  {
    // this is the sample mean minus mean[k]:
    const long double MmM = newSSum[k]/count;
    // mean[k] = (1-WS)*mean[k] + WS * sample_mean, which becomes:
    mean[k] = mean[k] + WS * MmM;
    // if WS==1 then varS is exact, otherwise update second moment
    // centered around current mean[k] (ie. E[(Sk-mean[k])^2])
    long double varS = newSSqSum[k]/count - MmM*MmM*(2*WS-WS*WS);
    if(varS < numeric_limits<long double>::epsilon()) varS = 1;
    std[k] = (1-WS) * std[k] + WS * std::sqrt(varS);
    invstd[k] = 1/(std[k]+EPS);
  }

  #ifndef NDEBUG
    if(learn_rank == 0) {
     ofstream outf("runningAverages.dat", ios::app);
     outf<<count<<" "<<1/invstd_reward<<" "<<print(mean)<<" "<<print(std)<<endl;
     outf.flush(); outf.close();
    }
    Uint cntSamp = 0;
    for(Uint i=0; i<Set.size(); i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions.load());
    if(WS>=1)
    {
      vector<long double> dbgStateSum(dimS,0), dbgStateSqSum(dimS,0);
      #pragma omp parallel
      {
        vector<long double> thr_dbgStateSum(dimS,0), thr_dbgStateSqSum(dimS,0);
        #pragma omp for schedule(dynamic)
        for(Uint i=0; i<Set.size(); i++)
          for(Uint j=0; j<Set[i]->ndata(); j++) {
            const auto S = standardize(Set[i]->tuples[j]->s);
            for(Uint k=0; k<dimS; k++) {
              thr_dbgStateSum[k] += S[k]; thr_dbgStateSqSum[k] += S[k]*S[k];
            }
          }
        #pragma omp critical
        for(Uint k=0; k<dimS; k++) {
          dbgStateSum[k]   += thr_dbgStateSum[k];
          dbgStateSqSum[k] += thr_dbgStateSqSum[k];
        }
      }
      for(Uint k=0; k<dimS; k++) {
        const Real dbgMean = dbgStateSum[k]/cntSamp;
        const Real dbgVar = dbgStateSqSum[k]/cntSamp - dbgMean*dbgMean;
        if(std::fabs(dbgMean)>.001 || std::fabs(dbgVar-1)>.001)
          cout <<k<<" mean:"<<dbgMean<<" std:"<<dbgVar<< endl;
      }
    }
  #endif
}

// Transfer a completed trajectory from the `inProgress` buffer to the data set
void MemoryBuffer::push_back(const int & agentId)
{
  if(inProgress[agentId]->tuples.size() > 2 ) //at least s0 and sT
  {
    inProgress[agentId]->finalize( readNSeenSeq() );

    nSeenSequences++;
    nCmplTransitions += inProgress[agentId]->ndata();

    pushBackSequence(inProgress[agentId]);
  }
  else
  {
    printf("Trashing %lu obs.\n",inProgress[agentId]->tuples.size());
    fflush(0);
    _dispose_object(inProgress[agentId]);
  }
  inProgress[agentId] = new Sequence();
}

void MemoryBuffer::prune(const FORGET ALGO, const Fval CmaxRho)
{
  //checkNData();
  assert(CmaxRho>=1);
  // vector indicating location of sequence to delete
  int old_ptr = -1, far_ptr = -1, dkl_ptr = -1, fit_ptr = -1, del_ptr = -1;
  Real dkl_val = -1, far_val = -1, fit_val = 9e9, old_ind = nSeenSequences;
  const int nB4 = Set.size(); const Fval invC = 1/CmaxRho;
  Real _nOffPol = 0, _totDKL = 0;
  #pragma omp parallel reduction(+ : _nOffPol, _totDKL)
  {
    pair<int,Real> farpol{-1,-1}, maxdkl{-1,-1}, minerr{-1,9e9}, oldest{-1,9e9};
    #pragma omp for schedule(dynamic)
    for(Uint i = 0; i < Set.size(); i++)
    {
      if(Set[i]->just_sampled >= 0)
      {
        Set[i]->nOffPol = 0; Set[i]->MSE = 0; Set[i]->sumKLDiv = 0;
        for(Uint j=0; j<Set[i]->ndata(); j++) {
          const Fval W = Set[i]->offPolicImpW[j];
          Set[i]->MSE += Set[i]->SquaredError[j];
          Set[i]->sumKLDiv += Set[i]->KullbLeibDiv[j];
          assert(Set[i]->SquaredError[j]>=0&&W>=0&&Set[i]->KullbLeibDiv[j]>=0);
          // sequence is off policy if offPol W is out of 1/C : C
          if(W>CmaxRho || W<invC) Set[i]->nOffPol += 1;
        }
        Set[i]->just_sampled = -1;
      }

      const Real W_MSE = Set[i]->MSE     /Set[i]->ndata();
      const Real W_FAR = Set[i]->nOffPol /Set[i]->ndata();
      const Real W_DKL = Set[i]->sumKLDiv/Set[i]->ndata();
      _nOffPol += Set[i]->nOffPol; _totDKL += Set[i]->sumKLDiv;

      if(Set[i]->ID<oldest.second) { oldest.second=Set[i]->ID; oldest.first=i; }
      if(    W_FAR >farpol.second) { farpol.second= W_FAR;     farpol.first=i; }
      if(    W_DKL >maxdkl.second) { maxdkl.second= W_DKL;     maxdkl.first=i; }
      if(    W_MSE <minerr.second) { minerr.second= W_MSE;     minerr.first=i; }
    }
    #pragma omp critical
    {
     if(oldest.second<old_ind) { old_ptr=oldest.first; old_ind=oldest.second; }
     if(farpol.second>far_val) { far_ptr=farpol.first; far_val=farpol.second; }
     if(maxdkl.second>dkl_val) { dkl_ptr=maxdkl.first; dkl_val=maxdkl.second; }
     if(minerr.second<fit_val) { fit_ptr=minerr.first; fit_val=minerr.second; }
    }
  }

  if(CmaxRho<=1) _nOffPol = 0; //then this counter and its effects are skipped
  nOffPol = _nOffPol; avgDKL = _totDKL/nTransitions.load();

  minInd = old_ind;
  assert( far_val <= 1 );
  assert( old_ptr < nB4 && far_ptr < nB4 && dkl_ptr < nB4 && fit_ptr < nB4 );
  assert( old_ptr >=  0 && far_ptr >=  0 && dkl_ptr >=  0 && fit_ptr >=  0 );
  switch(ALGO) {
      case OLDEST:     del_ptr = old_ptr; break;
      case FARPOLFRAC: del_ptr = far_ptr; break;
      case MAXKLDIV:   del_ptr = dkl_ptr; break;
      case MINERROR:   del_ptr = fit_ptr; break;
  }
  if(del_ptr<0) die(" ");

  // safety measures: do not delete trajectory if Nobs > Ntarget
  // but if N > Ntarget even if we remove the trajectory
  // done to avoid bugs if a sequence is longer than maxTotObsNum
  // negligible effect if hyperparameters are chosen wisely
  if(Set[old_ptr]->ID + (int)Set.size() < Set[del_ptr]->ID) del_ptr = old_ptr;
  if(nTransitions.load()-Set[del_ptr]->ndata() > maxTotObsNum) {
    std::swap(Set[del_ptr], Set.back());
    popBackSequence();
    needs_pass = true;
    prefixSum();
  }
  nPruned += nB4-Set.size();
  #ifdef PRIORITIZED_ER
   if( stepSinceISWeep++ >= 10 || needs_pass )
     updateImportanceWeights(); needs_pass = false; stepSinceISWeep = 0;
  #endif
}

void MemoryBuffer::prefixSum()
{
  vector<Uint> thdStarts(nThreads, 0);
  #pragma omp parallel num_threads(nThreads)
  {
    Uint locPrefix = 0;
    const int thrI = omp_get_thread_num();
    const Uint stride = std::ceil( Set.size() / (Real) nThreads );
    const Uint start = thrI*stride;
    const Uint end = std::min( (thrI+1)*stride, (Uint) Set.size());
    for(Uint i=start; i<end; i++) {
      Set[i]->prefix = locPrefix;
      locPrefix += Set[i]->ndata();
    }
    thdStarts[thrI] = locPrefix;
    #pragma omp barrier
    Uint thrPrefix = 0;
    for(int i=0; i<thrI; i++) thrPrefix += thdStarts[i];
    for(Uint i=start; i<end; i++) Set[i]->prefix += thrPrefix;
  }
}

#ifdef PRIORITIZED_ER

#if 0 // rank based probability

static inline float Q_rsqrt( const float number )
{
	union { float f; uint32_t i; } conv;
	static constexpr float threehalfs = 1.5F;
	const float x2 = number * 0.5F;
	conv.f  = number;
	conv.i  = 0x5f3759df - ( conv.i >> 1 );
  // Uncomment to do 2 iterations:
  //conv.f  = conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
	return conv.f * ( threehalfs - ( x2 * conv.f * conv.f ) );
}

void MemoryBuffer::updateImportanceWeights()
{
  // we need to collect all errors and rank them by magnitude
  // how do we manage to do most of the work in one pass?
  // 1) gather errors along with index
  // 2) sort them by decreasing error
  // 3) compute inv sqrt of all errors, same sweep also get minP
  //const float EPS = numeric_limits<float>::epsilon();
  using USI = unsigned short;
  using TupEST = tuple<float, USI, USI>;
  const Uint nData = nTransitions.load();
  vector<TupEST> errors(nData);
  // 1)
  #pragma omp parallel for schedule(dynamic)
  for(Uint i=0; i<Set.size(); i++) {
    const auto err_i = errors.data() + Set[i]->prefix;
    for(Uint j=0; j<Set[i]->ndata(); j++)
      err_i[j] = std::make_tuple(Set[i]->SquaredError[j], (USI)i, (USI)j );
  }

  // 2)
  const auto isAbeforeB = [&] ( const TupEST& a, const TupEST& b) {
                          return std::get<0>(a) > std::get<0>(b); };
  #if 0
    __gnu_parallel::sort(errors.begin(), errors.end(), isAbeforeB);
  #else //approximate 2 pass sort
  vector<Uint> thdStarts(nThreads, 0);
  #pragma omp parallel num_threads(nThreads)
  {
    const int thrI = omp_get_thread_num();
    const Uint stride = std::ceil(nData / (Real) nThreads);
    // avoid cache thrashing: create new vector for second sort
    { // first each sorts one chunk
      const Uint start = thrI*stride;
      const Uint end = std::min( (thrI+1)*stride, nData);
      std::sort(errors.begin()+start, errors.begin()+end, isAbeforeB);
    }

    #pragma omp barrier

    // now each thread gets a quantile of partial sorts
    vector<TupEST> load_loc;
    load_loc.reserve(stride); // because we want to push back
    for(Uint t=0; t<nThreads; t++) {
      const Uint i = (t + thrI) % nThreads;
      const Uint start = i*stride, end = std::min( (i+1)*stride, nData);
      #pragma omp for schedule(static) nowait // equally divided
      for(Uint j=start; j<end; j++) load_loc.push_back( errors[j] );
    }
    const Uint locSize = load_loc.size();
    thdStarts[thrI] = locSize;
    std::sort(load_loc.begin(), load_loc.end(), isAbeforeB);

    #pragma omp barrier // wait all those thdStarts values

    Uint threadStart = 0;
    for(int i=0; i<thrI; i++) threadStart += thdStarts[i];
    for(Uint i=0; i<locSize; i++) errors[i+threadStart] = load_loc[i];
  }
  #endif
  //for(Uint i=0; i<errors.size(); i++) cout << std::get<0>(errors[i]) << endl;

  // 3)
  float minP = 1e9;
  vector<float> probs = vector<float>(nData, 1);
  #pragma omp parallel for reduction(min:minP) schedule(static)
  for(Uint i=0; i<nData; i++) {
    // if samples never seen by optimizer the samples have high priority
    const float P = std::get<0>(errors[i])>0 ? Q_rsqrt(i+1) : 1;
    const Uint seq = get<1>(errors[i]), t = get<2>(errors[i]);
    probs[Set[seq]->prefix + t] = P;
    Set[seq]->priorityImpW[t] = P;
    minP = std::min(minP, P);
  }
  minPriorityImpW = minP;

  distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
}
#else // error based probability
void MemoryBuffer::updateImportanceWeights()
{
  const float EPS = numeric_limits<float>::epsilon();
  const Uint nData = nTransitions.load();
  vector<float> probs = vector<float>(nData, 1);
  float minP = 1e9, maxP = 0;

  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(Uint i=0; i<Set.size(); i++) {
    const auto ndata = Set[i]->ndata();
    const auto probs_i = probs.data() + Set[i]->prefix;

    for(Uint j=0; j<ndata; j++) {
      const float deltasq = (float)Set[i]->SquaredError[j];
      // do sqrt(delta^2)^alpha with alpha = 0.5
      const float P = deltasq>0.0f? std::sqrt(std::sqrt(deltasq+EPS)) : 0.0f;
      const float Pe = P + EPS, Qe = (P>0.0f? P : 1.0e9f) + EPS;
      minP = std::min(minP, Qe); // avoid nans in impW
      maxP = std::max(maxP, Pe);

      Set[i]->priorityImpW[j] = P;
      probs_i[j] = P;
    }
  }
  //for(Uint i=0; i<probs.size(); i++) cout << probs[i]<< endl;
  //cout <<minP <<" " <<maxP<<endl;
  // if samples never seen by optimizer the samples have high priority
  #pragma omp parallel for schedule(static)
  for(Uint i=0; i<nData; i++) if(probs[i]<=0) probs[i] = maxP;
  minPriorityImpW = minP;
  maxPriorityImpW = maxP;
  // std::discrete_distribution handles normalizing by sum P
  distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
}
#endif

#endif

void MemoryBuffer::getMetrics(ostringstream& buff)
{
  Real avgR = 0;
  for(Uint i=0; i<Set.size(); i++) avgR += Set[i]->totR;

  real2SS(buff, invstd_reward*avgR/(Set.size()+1e-7), 7, 0);
  real2SS(buff, 1/invstd_reward, 6, 1);
  real2SS(buff, avgDKL, 6, 1);

  buff<<" "<<std::setw(5)<<nSequences.load();
  buff<<" "<<std::setw(7)<<nTransitions.load();
  buff<<" "<<std::setw(7)<<nSeenSequences.load();
  buff<<" "<<std::setw(8)<<nSeenTransitions.load();
  buff<<" "<<std::setw(7)<<minInd;
  buff<<" "<<std::setw(6)<<(int)nOffPol;

  nPruned=0;
}
void MemoryBuffer::getHeaders(ostringstream& buff)
{
  buff <<
  "| avgR | stdr | DKL | nEp |  nObs | totEp | totObs | oldEp |nFarP ";
}

void MemoryBuffer::save(const string base, const Uint nStep, const bool bBackup)
{
  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
  fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
  fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
  fwrite(&invstd_reward, sizeof(Real),             1, wFile);
  fflush(wFile); fclose(wFile);

  if(bBackup) {
    ostringstream S; S<<std::setw(9)<<std::setfill('0')<<nStep;
    wFile = fopen((base+"scaling_"+S.str()+".raw").c_str(), "wb");
    fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
    fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
    fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
    fwrite(&invstd_reward, sizeof(Real),             1, wFile);
    fflush(wFile); fclose(wFile);
  }
}

void MemoryBuffer::restart(const string base)
{
  {
    FILE * wFile = fopen((base+"scaling.raw").c_str(), "rb");
    if(wFile == NULL) {
      printf("Parameters restart file %s not found.\n", (base+".raw").c_str());
      return;
    } else {
      printf("Restarting from file %s.\n", (base+"scaling.raw").c_str());
      fflush(0);
    }

    size_t size1 = fread(   mean.data(), sizeof(memReal),   mean.size(), wFile);
    size_t size2 = fread( invstd.data(), sizeof(memReal), invstd.size(), wFile);
    size_t size3 = fread(    std.data(), sizeof(memReal),    std.size(), wFile);
    size_t size4 = fread(&invstd_reward, sizeof(Real),             1, wFile);
    fclose(wFile);
    if(size1!=mean.size()|| size2!=invstd.size()|| size3!=std.size()|| size4!=1)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }
}

// number of returned samples depends on size of seq! (== to that of trans)
void MemoryBuffer::sampleTransition(Uint& seq, Uint& obs, const int thrID) {
  #ifndef PRIORITIZED_ER
    std::uniform_int_distribution<int> distObs(0, readNData()-1);
    const Uint ind = distObs(generators[thrID]);
  #else
    const Uint ind = distPER(generators[thrID]);
  #endif
  indexToSample(ind, seq, obs);
}

void MemoryBuffer::sampleSequence(Uint& seq, const int thrID) {
  //#ifndef PRIORITIZED_ER
    std::uniform_int_distribution<int> distSeq(0, readNSeq()-1);
    seq = distSeq(generators[thrID]);
  //#else
  //  seq = distPER(generators[thrID]);
  //#endif
}

void MemoryBuffer::sampleTransitions(vector<Uint>& seq, vector<Uint>& obs) {
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  #ifndef PRIORITIZED_ER
    std::uniform_int_distribution<Uint> distObs(0, readNData()-1);
  #else
    discrete_distribution<Uint> & distObs = distPER;
  #endif

  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(generators[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  // go through each element of ret to find corresponding seq and obs
  for (Uint k = 0, cntO = 0, i = 0; k<Set.size(); k++) {
    while(1) {
      assert(ret[i] >= cntO && i < seq.size());
      if(ret[i] < cntO + Set[k]->ndata()) { // is ret[i] in sequence k?
        obs[i] = ret[i] - cntO; // if ret[i]==cntO then obs 0 of k and so forth
        seq[i] = k;
        i++; // next iteration remember first i-1 were already found
      }
      else break;
      if(i == seq.size()) break; // then found all elements of sequence k
    }
    assert(cntO == Set[k]->prefix);
    if(i == seq.size()) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == Set.size()) die(" "); // at last iter we must have found all
  }
}

void MemoryBuffer::sampleSequences(vector<Uint>& seq) {
  const Uint N = seq.size();
  if( readNSeq() > N*5 ) {
    for(Uint i=0; i<N; i++) sampleSequence(seq[i], 0);
  } else { // if N is large, make sure we do not repeat indices
    seq.resize(readNSeq());
    std::iota(seq.begin(), seq.end(), 0);
    std::shuffle(seq.begin(), seq.end(), generators[0]);
    seq.resize(N);
  }
  const auto compare = [&](Uint a, Uint b) {
    return Set[a]->ndata() > Set[b]->ndata();
  };
  std::sort(seq.begin(), seq.end(), compare);
}

void MemoryBuffer::indexToSample(const int nSample,Uint& seq,Uint& obs) const {
  int k = 0, back = 0, indT = Set[0]->ndata();
  while (nSample >= indT) {
    assert(k+2<=(int)Set.size());
    back = indT;
    indT += Set[++k]->ndata();
  }
  assert(nSample>=back && Set[k]->ndata()>(Uint)nSample-back);
  seq = k; obs = nSample-back;
}
