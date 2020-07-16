//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "MemoryBuffer.h"
#include "Sampling.h"
#include <algorithm>
//#include <parallel/algorithm>

namespace smarties
{

Sampling::Sampling(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq)
: gens(G), RM(R), episodes(RM->episodes), bSampleEpisodes(bSeq) {}

long Sampling::nEpisodes() const { return RM->nStoredEps(); }
long Sampling::nTransitions() const { return RM->nStoredSteps(); }
void Sampling::setMinMaxProb(const Real maxP, const Real minP) {
    RM->minPriorityImpW = minP;
    RM->maxPriorityImpW = maxP;
}

void Sampling::IDtoSeqStep(std::vector<Uint>& seq, std::vector<Uint>& obs,
  const std::vector<Uint>& sampledTsteps, const Uint nSeqs)
{
  // go through each element of sampledTsteps to find corresponding seq and obs
  const Uint Bsize = seq.size();
  assert(obs.size() == Bsize and sampledTsteps.size() == Bsize);

  for (Uint k=0, i=0; k < nSeqs && i < Bsize; ++k)
  {
    assert(sampledTsteps[i] >= episodes[k].prefix);
    while (i < Bsize and
           sampledTsteps[i] < episodes[k].prefix + episodes[k].ndata() )
    {
      // if ret[i]==prefix then obs 0 of k and so forth:
      obs[i] = sampledTsteps[i] - episodes[k].prefix;
      seq[i] = k;
      ++i; // next iteration remember first i-1 were already found
    }
    assert(k+1 < nSeqs or i == Bsize); // at last iter we must have found all
  }
}

void Sampling::IDtoSeqStep_par(std::vector<Uint>& seq, std::vector<Uint>& obs,
  const std::vector<Uint>& ret, const Uint nSeqs)
{ // go through each element of ret to find corresponding seq and obs
  #pragma omp parallel
  {
    Uint i = 0;
    #pragma omp for schedule(static)
    for (Uint k=0; k<nSeqs; ++k) {
      // sample i lies before start of episode k:
      while(i < ret.size() && ret[i] < episodes[k].prefix) ++i;

      while(i < ret.size() && ret[i] < episodes[k].prefix + episodes[k].ndata())
      {
        // if ret[i]==prefix then obs 0 of k and so forth:
        obs[i] = ret[i] - episodes[k].prefix;
        seq[i] = k;
        ++i; // next iteration remember first i-1 were already found
      }
    }
  }
}

void Sampling::updatePrefixes()
{
  const long nSeqs = nEpisodes();
  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    episodes[i].prefix = locPrefix;
    locPrefix += episodes[i].ndata();
  }
}

void Sampling::checkPrefixes()
{
  #ifndef NDEBUG
    const long nSeqs = nEpisodes(), nData = nTransitions();
    assert(episodes.size() == (size_t) nSeqs);
    for(long i=0, locPrefix=0; i<nSeqs; ++i) {
      assert(episodes[i].prefix == (Uint) locPrefix);
      locPrefix += episodes[i].ndata();
      if(i+1 == nSeqs) assert(locPrefix == nData);
    }
  #endif
}

Sample_uniform::Sample_uniform(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_uniform::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  assert(seq.size() == obs.size());
  //std::unique_lock<std::mutex> lock(RM->dataset_mutex, std::defer_lock);
  //lock.lock();
  const long nBatch = obs.size();
  //lock.unlock();

  #ifndef NDEBUG
  {
    std::lock_guard<std::mutex> lock(RM->dataset_mutex);
    checkPrefixes();
  }
  #endif

  if(bSampleEpisodes)
  {
    const long nSeqs = nEpisodes();
    std::uniform_int_distribution<Uint> distSeq(0, nSeqs-1);
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return distSeq(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }

    const auto compare = [&](const Uint a, const Uint b) {
      return episodes[a].ndata() > episodes[b].ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (long i=0; i<nBatch; ++i) obs[i] = episodes[seq[i]].ndata() - 1;
  }
  else
  {
    const long nData = nTransitions();
    std::uniform_int_distribution<Uint> distObs(0, nData-1);
    std::vector<Uint> ret(nBatch);
    std::vector<Uint>::iterator it = ret.begin();
    while(it not_eq ret.end())
    {
      std::generate(it, ret.end(), [&]() { return distObs(gens[0]); } );
      std::sort(ret.begin(), ret.end());
      it = std::unique (ret.begin(), ret.end());
    } // ret is now also sorted!
    const auto nSeq = nEpisodes();
    IDtoSeqStep(seq, obs, ret, nSeq);
    #if 0
    auto obs2 = obs, seq2 = seq;
    IDtoSeqStep_par(seq2, obs2, ret, nSeq);
    if (! std::equal(obs.begin(), obs.end(), obs2.begin()) )
       die("obs obs2 are not equal");
    if (! std::equal(seq.begin(), seq.end(), seq2.begin()) )
       die("seq seq2 are not equal");
    #endif
  }
}
void Sample_uniform::prepare(std::atomic<bool>& needs_pass)
{
  if (needs_pass)
  {
    std::lock_guard<std::mutex> lock(RM->dataset_mutex);
    updatePrefixes();
    needs_pass = false;
  }
  #ifndef NDEBUG
    else checkPrefixes();
  #endif
}
bool Sample_uniform::requireImportanceWeights() { return false; }


Sample_unifEps::Sample_unifEps(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_unifEps::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  assert(seq.size() == obs.size());
  const Uint N = obs.size(), nSeqs = nEpisodes();
  std::uniform_int_distribution<Uint> distSeq(0, nSeqs-1);
  std::vector<Uint>::iterator it = seq.begin();
  while(it not_eq seq.end()) {
    std::generate(it, seq.end(), [&]() { return distSeq(gens[0]); } );
    std::sort( seq.begin(), seq.end() );
    if (N > nSeqs) break;
    it = std::unique( seq.begin(), seq.end() );
  }
  if(bSampleEpisodes) {
    for (Uint i=0; i<N; ++i) obs[i] = episodes[seq[i]].ndata()-1;
  } else {
    std::uniform_real_distribution<Real> dist(0, 1);
    for (Uint i=0; i<N; ++i) obs[i] = dist(gens[0]) * episodes[seq[i]].ndata();
  }
}
void Sample_unifEps::prepare(std::atomic<bool>& needs_pass)
{
}
bool Sample_unifEps::requireImportanceWeights() { return false; }


Sample_impLen::Sample_impLen(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_impLen::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");
  const Uint nBatch = obs.size();

  if(bSampleEpisodes)
  {
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return dist(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }
    const auto compare = [&](const Uint a, const Uint b) {
      return episodes[a].ndata() > episodes[b].ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = episodes[seq[i]].ndata() - 1;
  }
  else
  {
    std::uniform_real_distribution<float> distStep(0, 1);
    std::vector<std::pair<Uint, Uint>> S (nBatch);
    std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
    while(it not_eq S.end())
    {
      std::generate(it, S.end(), [&] () {
          const Uint seqID = dist(gens[0]);
          const Uint stepID = distStep(gens[0]) * episodes[seqID].ndata();
          return std::pair<Uint, Uint> {seqID, stepID};
        }
      );
      std::sort( S.begin(), S.end() );
      it = std::unique( S.begin(), S.end() );
    }
    for (Uint i=0; i<nBatch; ++i) {
      seq[i] = S[i].first;
      obs[i] = S[i].second;
    }
  }
}
void Sample_impLen::prepare(std::atomic<bool>& needs_pass)
{
  if(not needs_pass) return;
  needs_pass = false;
  const Uint nSeqs = nEpisodes();
  std::vector<float> probs(nSeqs, 0);

  #pragma omp parallel for schedule(static)
  for (Uint i=0; i<nSeqs; ++i) probs[i] = episodes[i].ndata();

  dist = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}
bool Sample_impLen::requireImportanceWeights() { return false; }


TSample_shuffle::TSample_shuffle(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_shuffle::prepare(std::atomic<bool>& needs_pass)
{
  if(not needs_pass) return;
  needs_pass = false;

  const long nSeqs = nEpisodes(), nData = nTransitions();
  samples.resize(nData);

  updatePrefixes();

  #pragma omp parallel for schedule(dynamic)
  for(long i = 0; i < nSeqs; ++i)
    for(Uint j=0, k=episodes[i].prefix; j<episodes[i].ndata(); ++j, ++k)
      samples[k] = std::pair<unsigned, unsigned>{i, j};

  //const auto RNG = [&](const int max) {
  //  assert(max > 0);
  //  std::uniform_int_distribution<int> dist(0, max-1);
  //  return dist(gens[0]);
  //};
  //__gnu_parallel::random_shuffle(samples.begin(), samples.end(), RNG);
  //std::random_shuffle(samples.begin(), samples.end(), RNG);
  std::shuffle(samples.begin(), samples.end(), gens[0]);
}
void TSample_shuffle::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  for(Uint i=0; i<seq.size(); ++i)
  {
    seq[i] = samples.back().first;
    obs[i] = samples.back().second;
    samples.pop_back();
  }
}
bool TSample_shuffle::requireImportanceWeights() { return false; }


TSample_impRank::TSample_impRank(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_impRank::prepare(std::atomic<bool>& needs_pass)
{
  if( needs_pass == false and stepSinceISWeep++ < 10 ) return;

  stepSinceISWeep = 0;
  needs_pass = false;
  // we need to collect all errors and rank them by magnitude
  // how do we manage to do most of the work in one pass?
  // 1) gather errors along with index
  // 2) sort them by decreasing error
  // 3) compute inv sqrt of all errors, same sweep also get minP
  //const float EPS = numeric_limits<float>::epsilon();
  using TupEST = std::tuple<float, unsigned, unsigned>;
  const unsigned nSeqs = nEpisodes(), nData = nTransitions();

  std::vector<TupEST> errors(nData);
  // 1)
  for(unsigned i=0, locPrefix=0; i<nSeqs; ++i) {
    auto & EP = episodes[i];
    EP.prefix = locPrefix;
    const unsigned epNsteps = EP.ndata();
    const auto err_i = errors.data() + locPrefix;
    locPrefix += epNsteps;
    for(unsigned j=0; j<epNsteps; ++j)
      err_i[j] = std::make_tuple(EP.SquaredError(j), i, j);
  }

  // 2)
  const auto isAbeforeB = [&] ( const TupEST& a, const TupEST& b) {
                          return std::get<0>(a) > std::get<0>(b); };
  //__gnu_parallel
  std::sort(errors.begin(), errors.end(), isAbeforeB);
  //for(Uint i=0; i<errors.size(); ++i) cout << std::get<0>(errors[i]) << endl;

  // 3)
  float minP = 1e9;
  std::vector<float> probs = std::vector<float>(nData, 1);
  #pragma omp parallel for reduction(min:minP) schedule(static)
  for(long i=0; i<nData; ++i) {
    // if samples never seen by optimizer the samples have high priority
    //const float P = std::get<0>(errors[i])>0 ? approxRsqrt(i+1) : 1;
    const float P = std::get<0>(errors[i])>0 ? 1.0/std::cbrt(i+1) : 1;
    const unsigned seq = std::get<1>(errors[i]), t = std::get<2>(errors[i]);
    auto & EP = episodes[seq];
    probs[EP.prefix + t] = P;
    EP.priorityImpW[t] = P;
    minP = std::min(minP, P);
  }

  setMinMaxProb(1, minP);
  distObs = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}

void TSample_impRank::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  const long nSeqs = nEpisodes();
  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(gens[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  IDtoSeqStep(seq, obs, ret, nSeqs);
}
bool TSample_impRank::requireImportanceWeights() { return true; }



TSample_impErr::TSample_impErr(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_impErr::prepare(std::atomic<bool>& needs_pass)
{
  if( ( stepSinceISWeep++ >= 10 || needs_pass ) == false ) return;
  stepSinceISWeep = 0;
  needs_pass = false;

  const float EPS = std::numeric_limits<float>::epsilon();
  const long nSeqs = nEpisodes(), nData = nTransitions();
  std::vector<float> probs = std::vector<float>(nData, 1);

  updatePrefixes();

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(long i=0; i<nSeqs; ++i)
  {
    const Uint ndata = episodes[i].ndata();
    const auto probs_i = probs.data() + episodes[i].prefix;

    for(Uint j=0; j<ndata; ++j)
    {
      const float deltasq = (float) episodes[i].SquaredError(j);
      // do sqrt(delta^2)^alpha with alpha = 0.5
      const float P = deltasq>0.0f? std::sqrt(std::sqrt(deltasq+EPS)) : 0.0f;
      const float Pe = P + EPS;
      const float Qe = deltasq>0.0f? P + EPS : 1.0e9f ; // avoid nans in impW
      minP = std::min(minP, Qe);
      maxP = std::max(maxP, Pe);
      episodes[i].priorityImpW[j] = P;
      probs_i[j] = P;
    }
  }
  //for(Uint i=0; i<probs.size(); ++i) cout << probs[i]<< endl;
  //cout <<minP <<" " <<maxP<<endl;
  // if samples never seen by optimizer the samples have high priority
  #pragma omp parallel for schedule(static)
  for(long i=0; i<nData; ++i) if(probs[i]<=0) probs[i] = maxP;

  setMinMaxProb(maxP, minP);
  // std::discrete_distribution handles normalizing by sum P
  distObs = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}
void TSample_impErr::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  const long nSeqs = nEpisodes();
  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(gens[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  IDtoSeqStep(seq, obs, ret, nSeqs);
}
bool TSample_impErr::requireImportanceWeights() { return true; }



Sample_impSeq::Sample_impSeq(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_impSeq::prepare(std::atomic<bool>& needs_pass)
{
  if( stepSinceISWeep++ < 5 && not needs_pass ) return;
  stepSinceISWeep = 0;
  needs_pass = false;

  const float EPS = std::numeric_limits<float>::epsilon();
  const long nSeqs = nEpisodes();
  std::vector<float> probs = std::vector<float>(nSeqs, 1);

  Fval maxMSE = 0;
  for(long i=0; i<nSeqs; ++i)
    maxMSE = std::max(maxMSE, episodes[i].sumSquaredErr / episodes[i].ndata());

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(long i=0; i<nSeqs; ++i)
  {
    auto & EP = episodes[i];
    const Uint ndata = EP.ndata();
    float sumErrors = EP.sumSquaredErr;
    for(Uint j=0; j<ndata; ++j)
      if( EP.SquaredError(j) <= 0 ) sumErrors += maxMSE;
    //sampling P is episode's RMSE to the power beta=.5 times length of episode
    const float P = std::sqrt( std::sqrt( (sumErrors + EPS)/ndata ) ) * ndata;
    std::fill(EP.priorityImpW.begin(), EP.priorityImpW.end(), P);
    minP = std::min(minP, P); // avoid nans in impW
    maxP = std::max(maxP, P);
    probs[i] = P;
  }

  setMinMaxProb(maxP, minP);

  // std::discrete_distribution handles normalizing by sum P
  distObs = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}
void Sample_impSeq::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");
  const Uint nBatch = obs.size();

  if(bSampleEpisodes)
  {
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return distObs(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }
    const auto compare = [&](const Uint a, const Uint b) {
      return episodes[a].ndata() > episodes[b].ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = episodes[seq[i]].ndata() - 1;
  }
  else
  {
    std::uniform_real_distribution<float> distT(0, 1);
    std::vector<std::pair<Uint, Uint>> S (nBatch);

    std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
    while(it not_eq S.end()) {
      std::generate(it, S.end(), [&] () {
         const Uint _s = distObs(gens[0]);
         const Uint _t = distT(gens[0]) * episodes[_s].ndata();
         return std::pair<Uint, Uint> {_s, _t};
        }
      );
      std::sort( S.begin(), S.end() );
      it = std::unique( S.begin(), S.end() );
    }

    for (Uint i=0; i<nBatch; ++i) { seq[i] = S[i].first; obs[i] = S[i].second; }
  }
}
bool Sample_impSeq::requireImportanceWeights() { return true; }

std::unique_ptr<Sampling> Sampling::prepareSampler(MemoryBuffer* const R,
                                                   HyperParameters & S,
                                                   ExecutionInfo & D)
{
  std::unique_ptr<Sampling> ret = nullptr;

  if(S.dataSamplingAlgo == "uniform") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: uniform probability.\n");
    ret = std::make_unique<Sample_uniform>(D.generators,R,S.bSampleEpisodes);
    //ret = std::make_unique<Sample_unifEps>(D.generators,R,S.bSampleEpisodes);
  }

  if(S.dataSamplingAlgo == "impLen") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: "
           "probability is proportional to episode-length.%s\n",
           S.bSampleEpisodes? "" : " Equivalent to uniform probability.");
    ret = std::make_unique<Sample_impLen>(D.generators,R,S.bSampleEpisodes);
  }

  if(S.dataSamplingAlgo == "shuffle") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: "
           "shuffle-based uniform probability.\n");
    ret = std::make_unique<TSample_shuffle>(D.generators,R,S.bSampleEpisodes);
    if(S.bSampleEpisodes) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERrank") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: "
           "rank based prioritized replay.\n");
    ret = std::make_unique<TSample_impRank>(D.generators,R,S.bSampleEpisodes);
    if(S.bSampleEpisodes) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERerr") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: "
           "error value based prioritized replay.\n");
    ret = std::make_unique<TSample_impErr>(D.generators,R,S.bSampleEpisodes);
    if(S.bSampleEpisodes) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERseq") {
    if(D.world_rank == 0)
    printf("Experience Replay sampling algorithm: "
           "episodes' mean squared error based prioritized replay.\n");
    ret = std::make_unique<Sample_impSeq>(D.generators,R,S.bSampleEpisodes);
  }

  assert(ret not_eq nullptr);
  return ret;
}

}
