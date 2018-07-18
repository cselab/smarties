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
 policyVecDim(_s.policyVecDim), sI(env->sI), aI(env->aI), _agents(env->agents),
 generators(_s.generators), mean(sI.inUseMean()), invstd(sI.inUseInvStd()),
 std(sI.inUseStd()), learn_rank(_s.learner_rank), learn_size(_s.learner_size),
 gamma(_s.gamma) {
  assert(_s.nAgents>0);
  inProgress.resize(_s.nAgents);
  for (int i=0; i<_s.nAgents; i++) inProgress[i] = new Sequence();
  gen = new Gen(&generators[0]);
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
  if(bWriteToFile) a.writeData(learn_rank, pol);
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
  if(bWriteToFile) a.writeData(learn_rank, Rvec(policyVecDim, 0));
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
          const Real sk = Set[i]->tuples[j]->s[k] - mean[k];
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
    Real varR = newstdvr/count;
    if(varR < numeric_limits<Real>::epsilon())
       varR = numeric_limits<Real>::epsilon();
    invstd_reward = (1-WR)*invstd_reward +WR/(std::sqrt(varR)+EPS);
  }
  for(Uint k=0; k<dimS && WS>0; k++)
  {
    // this is the sample mean minus mean[k]:
    const Real MmM = newSSum[k]/count;
    // mean[k] = (1-WS)*mean[k] + WS * sample_mean, which becomes:
    mean[k] = mean[k] + WS * MmM;
    // if WS==1 then varS is exact, otherwise update second moment
    // centered around current mean[k] (ie. E[(Sk-mean[k])^2])
    Real varS = newSSqSum[k]/count - MmM*MmM*(2*WS-WS*WS);
    if(varS < numeric_limits<Real>::epsilon())
       varS = numeric_limits<Real>::epsilon();
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
    if(WS>0)
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

void MemoryBuffer::prune(const FORGET ALGO, const Real CmaxRho)
{
  //checkNData();
  assert(CmaxRho>=1);
  // vector indicating location of sequence to delete
  int old_ptr = -1, far_ptr = -1, dkl_ptr = -1, fit_ptr = -1, del_ptr = -1;
  Real dkl_val = -1, far_val = -1, fit_val = 9e9, old_ind = nSeenSequences;
  const int nB4 = Set.size(); const Real invC = 1/CmaxRho;
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
          const Real W = Set[i]->offPolicImpW[j];
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
      die(" ");
  }

  // safety measures: do not delete trajectory if Nobs > Ntarget
  // but if N > Ntarget even if we remove the trajectory
  // done to avoid bugs if a sequence is longer than maxTotObsNum
  // negligible effect if hyperparameters are chosen wisely
  if(Set[old_ptr]->ID + (int)Set.size() < Set[del_ptr]->ID) del_ptr = old_ptr;
  if(nTransitions.load()-Set[del_ptr]->ndata() > maxTotObsNum) {
    std::swap(Set[del_ptr], Set.back());
    popBackSequence();
  }
  nPruned += nB4-Set.size();

  #ifdef IMPORTSAMPLE
    updateImportanceWeights();
  #endif
}

void MemoryBuffer::updateImportanceWeights()
{
  /*
  Rvec probs(nTransitions.load()), wghts(nTransitions.load());
  const Real EPS = numeric_limits<float>::epsilon();
  Real minP = 1e9, sumP = 0;
  #pragma omp parallel reduction(min: minP) reduction(+: sumP)
  for(Uint i=0, k=0; i<Set.size(); i++) {
    #pragma omp for nowait
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      const Real P = Set[i]->SquaredError[j]*Set[i]->offPolicImpW[j] + EPS;
      minP  = std::min(minP, P);
      sumP += P;
      probs[k+j] = P;
    }
    k += Set[i]->ndata();
  }

  #pragma omp parallel
  for(Uint i=0, k=0; i<Set.size(); i++) {
    #pragma omp for nowait
    for(Uint j=0; j<Set[i]->ndata(); j++) {
      wghts[k+j] = minP / probs[k+j];
      probs[k+j] = probs[k+j] / sumP;
      Set[i]->priorityImpW[j] = wghts[k+j];
    }
    k += Set[i]->ndata();
  }

  if(dist not_eq nullptr) delete dist;
  dist = new std::discrete_distribution<Uint>(probs.begin(), probs.end());
  */
}

void MemoryBuffer::getMetrics(ostringstream& buff)
{
  Real avgR = 0;
  for(Uint i=0; i<Set.size(); i++) avgR += Set[i]->totR;

  real2SS(buff, invstd_reward*avgR/Set.size(), 7, 0);
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

void MemoryBuffer::save(const string base, const Uint nStep)
{
  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  fwrite(   mean.data(), sizeof(Real),   mean.size(), wFile);
  fwrite( invstd.data(), sizeof(Real), invstd.size(), wFile);
  fwrite(    std.data(), sizeof(Real),    std.size(), wFile);
  fwrite(&invstd_reward, sizeof(Real),             1, wFile);
  fflush(wFile); fclose(wFile);

  if(nStep % FREQ_BACKUP == 0 && nStep > 0) {
    ostringstream S; S<<std::setw(9)<<std::setfill('0')<<nStep;
    wFile = fopen((base+"scaling_"+S.str()+".raw").c_str(), "wb");
    fwrite(   mean.data(), sizeof(Real),   mean.size(), wFile);
    fwrite( invstd.data(), sizeof(Real), invstd.size(), wFile);
    fwrite(    std.data(), sizeof(Real),    std.size(), wFile);
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

    size_t size1 = fread(   mean.data(), sizeof(Real),   mean.size(), wFile);
    size_t size2 = fread( invstd.data(), sizeof(Real), invstd.size(), wFile);
    size_t size3 = fread(    std.data(), sizeof(Real),    std.size(), wFile);
    size_t size4 = fread(&invstd_reward, sizeof(Real),             1, wFile);
    fclose(wFile);
    if(size1!=mean.size()|| size2!=invstd.size()|| size3!=std.size()|| size4!=1)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }
  return;

  const Uint writesize = 3 +sI.dim +aI.dim +policyVecDim;
  int agentID = 0, info = 0, sampID = 0;
  Rvec policy(policyVecDim), action(aI.dim), state(sI.dim);
  char cpath[256];

  while (true) {
    sprintf(cpath, "obs_rank%02d_agent%03d.raw", learn_rank, agentID);
    FILE*pFile = fopen(cpath, "rb");
    if(pFile==NULL) { printf("Couldnt open file %s.\n", cpath); break; }

    float* buf = (float*) malloc(writesize*sizeof(float));
    while(true) {
      size_t ret = fread(buf, sizeof(float), writesize, pFile);
      if (ret == 0) break;
      if (ret != writesize) _die("Error reading datafile %s", cpath);
      Uint k = 0;
      info = buf[k++]; sampID = buf[k++];

      if((sampID==0) != (info==1)) die("Mismatch in transition counter\n");
      if(sampID!=_agents[0]->transitionID+1 && info!=1) die(" transitionID");

      for(Uint i=0; i<sI.dim; i++) state[i]  = buf[k++];
      for(Uint i=0; i<aI.dim; i++) action[i] = buf[k++];
      Real reward = buf[k++];
      for(Uint i=0; i<policyVecDim; i++) policy[i] = buf[k++];
      assert(k == writesize);

      _agents[0]->update(info, state, reward);
      add_state(*_agents[0]);
      inProgress[0]->add_action(action, policy);
      if(info == 2) push_back(0);
    }
    if(_agents[0]->getStatus() not_eq 2) push_back(0); //(agentID is 0)
    fclose(pFile); free(buf);
    agentID++;
  }
  if(agentID==0) { printf("Couldn't restart transition data.\n"); } //return 1;
  //push_back(0);
  printf("Found %d broken seq out of %d/%d.\n",
    nBroken.load(),nSequences.load(),nTransitions.load());
  //return 0;
}

// number of returned samples depends on size of seq! (== to that of trans)
void MemoryBuffer::sampleTransition(Uint& seq, Uint& obs, const int thrID)
{
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<int> distObs(0, readNData()-1);
    const Uint ind = distObs(generators[thrID]);
  #else
    const Uint ind = (*dist)(generators[thrID]);
  #endif
  indexToSample(ind, seq, obs);
}

void MemoryBuffer::sampleSequence(Uint& seq, const int thrID)
{
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<int> distSeq(0, readNSeq()-1);
    seq = distSeq(generators[thrID]);
  #else
    seq = (*dist)(generators[thrID]);
  #endif
}

void MemoryBuffer::sampleTransitions_OPW(vector<Uint>&seq, vector<Uint>&obs)
{
  assert(seq.size() == obs.size());
  vector<Uint> s = seq, o = obs;
  int nThr_loc = nThreads;
  while(seq.size() % nThr_loc) nThr_loc--;
  const int stride = seq.size()/nThr_loc, N = seq.size();
  assert(nThr_loc>0 && N % nThr_loc == 0 && stride > 0);
  // Perf tweak: sort by offPol weight to aid load balance of algos like
  // Racer/PPO. "Far Policy" obs are discarded due to opcW being out of range
  // and will trigger a resampling. Sorting here affects tasking order.
  const auto isAbeforeB = [&] (const pair<Uint,Real> a, const pair<Uint,Real> b)
  { return a.second > b.second; };

  vector<pair<Uint, Real>> load(N);
  #pragma omp parallel num_threads(nThr_loc)
  {
    const int thrI = omp_get_thread_num(), start = thrI*stride;
    assert(nThr_loc == omp_get_num_threads());

    sampleMultipleTrans(&s[start], &o[start], stride, thrI);
    for(int i=start; i<start+stride; i++) {
      const Real W = Set[s[i]]->offPolicImpW[o[i]], invW = 1/W;
      load[i].first = i; load[i].second = std::max(W, invW);
    }
    std::sort(load.begin()+start, load.begin()+start+stride, isAbeforeB);

    // additional parallel partial sort if batchsize makes it worthwhile
    if(stride % nThr_loc == 0) {
      vector<pair<Uint, Real>> load_loc(stride);
      const int preSortChunk = stride / nThr_loc;

      #pragma omp barrier

      for(int i=0, k=0; i<nThr_loc; i++) // avoid cache thrashing
       for(int j=0; j<preSortChunk; j++)
         load_loc[k++] = load[thrI*preSortChunk + i*stride + j];

      std::sort(load_loc.begin(), load_loc.end(), isAbeforeB);

      #pragma omp barrier
      for(int i=0; i<stride; i++) load[i+start] = load_loc[i];
    }
  }
  //for (Uint i=0; i<seq.size(); i++) cout<<load[i].second<<endl; cout<<endl;
  std::sort(load.begin(), load.end(), isAbeforeB);
  for (Uint i=0; i<seq.size(); i++) {
    obs[i] = o[load[i].first];
    seq[i] = s[load[i].first];
  }
}

void MemoryBuffer::sampleTransitions(vector<Uint>&seq, vector<Uint>&obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<Uint> distObs(0, readNData()-1);
  #else
    Gen & distObs = *dist;
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
    if(i == seq.size()) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == Set.size()) die(" "); // at last iter we must have found all
  }
}

void MemoryBuffer::sampleSequences(vector<Uint>& seq)
{
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

void MemoryBuffer::indexToSample(const int nSample, Uint& seq, Uint& obs) const
{
  int k = 0, back = 0, indT = Set[0]->ndata();
  while (nSample >= indT) {
    assert(k+2<=(int)Set.size());
    back = indT;
    indT += Set[++k]->ndata();
  }
  assert(nSample>=back && Set[k]->ndata()>(Uint)nSample-back);
  seq = k; obs = nSample-back;
}

void MemoryBuffer::sampleMultipleTrans(Uint* seq, Uint* obs, const Uint N, const int thrID)
{
  vector<Uint> samples(N);

  #ifndef IMPORTSAMPLE
    std::uniform_int_distribution<Uint> distObs(0, readNData()-1);
    for (Uint k=0; k<N; k++) samples[k] = distObs(generators[thrID]);
  #else
    for (Uint k=0; k<N; k++) samples[k] = (*dist)(generators[thrID]);
  #endif

  std::sort(samples.begin(), samples.end());

  Uint cntO = 0, samp = 0;
  for (Uint k=0; k<Set.size(); k++) {
    for(Uint i=samp; i<N; i++) {
      if(samples[i] < cntO+Set[k]->ndata()) { // cannot sample last state in seq
        seq[i] = k;
        obs[i] = samples[i]-cntO;
        samp = i+1; // next iteration remember first samp-1 were already found
      }
    }
    if(samp == N) break;
    cntO += Set[k]->ndata();
    if(k+1 == Set.size()) assert(cntO == nTransitions.load() && samp == N);
  }
}
