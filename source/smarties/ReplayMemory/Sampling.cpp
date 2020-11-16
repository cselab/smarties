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
  const Uint batchsize = seq.size();
  assert(obs.size() == batchsize and sampledTsteps.size() == batchsize);

  for (Uint k=0, i=0, prefix=0; k < nSeqs && i < batchsize; ++k)
  {
    assert(sampledTsteps[i] >= prefix);
    const Uint nsteps = episodes[k]->ndata();
    while (i < batchsize and sampledTsteps[i] < prefix + nsteps)
    {
      // if ret[i]==prefix then obs 0 of k and so forth:
      obs[i] = sampledTsteps[i] - prefix;
      seq[i] = k;
      ++i; // next iteration remember first i-1 were already found
    }
    prefix += nsteps;
    assert(k+1 < nSeqs or i == batchsize); // at last iter we must have found all
  }
}

Sample_uniform::Sample_uniform(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_uniform::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  assert(seq.size() == obs.size());
  const Uint nBatch = obs.size();

  if(bSampleEpisodes)
  {
    const Uint nSeqs = nEpisodes();
    std::uniform_int_distribution<Uint> distSeq(0, nSeqs-1);
    if (nSeqs >= 2*nBatch)
    {
      std::vector<Uint>::iterator it = seq.begin();
      while(it not_eq seq.end())
      {
        std::generate(it, seq.end(), [&]() { return distSeq(gens[0]); } );
        std::sort( seq.begin(), seq.end() );
        it = std::unique( seq.begin(), seq.end() );
      }
    } else {
      seq.resize(nSeqs);
      std::iota(seq.begin(), seq.end(), 0);
      // if fewer episodes than the batchsize, fill with random duplicates:
      while (seq.size() < nBatch) seq.push_back( distSeq(gens[0]) );
      std::shuffle(seq.begin(), seq.end(), gens[0]);
      seq.resize(nBatch);
    }
    const auto compare = [&](const Uint a, const Uint b) {
      return episodes[a]->ndata() > episodes[b]->ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = episodes[seq[i]]->ndata() - 1;
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
  }
}
void Sample_uniform::prepare() {}
bool Sample_uniform::requireImportanceWeights() { return false; }

TSample_impRank::TSample_impRank(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_impRank::prepare()
{
  // we need to collect all errors and rank them by magnitude
  // how do we manage to do most of the work in one pass?
  // 1) gather errors along with index
  // 2) sort them by decreasing error
  // 3) compute inv sqrt of all errors, same sweep also get minP
  //const float EPS = numeric_limits<float>::epsilon();
  using TupEST = std::tuple<float, unsigned, unsigned>;
  const unsigned nSeqs = nEpisodes(), nData = nTransitions();

  std::vector<TupEST> errors(nData);
  std::vector<Uint> prefixes(nSeqs);
  // 1)
  for(Uint i=0, prefix=0; i<nSeqs; ++i) {
    const auto & EP = * episodes[i].get();
    prefixes[i] = prefix;
    const Uint epNsteps = EP.ndata();
    const auto err_i = errors.data() + prefix;
    prefix += epNsteps;
    for(Uint j=0; j<epNsteps; ++j)
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
  for(unsigned i=0; i<nData; ++i) {
    // if samples never seen by optimizer the samples have high priority
    //const float P = std::get<0>(errors[i])>0 ? approxRsqrt(i+1) : 1;
    const float P = std::get<0>(errors[i])>0 ? 1/std::sqrt(std::sqrt(i+1)) : 1;
    const Uint seq = std::get<1>(errors[i]), t = std::get<2>(errors[i]);
    episodes[seq]->priorityImpW[t] = P;
    probs[prefixes[seq] + t] = P;
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
void TSample_impErr::prepare()
{
  const float EPS = std::numeric_limits<float>::epsilon();
  const Uint nSeqs = nEpisodes(), nData = nTransitions();
  std::vector<float> probs = std::vector<float>(nData, 1);
  std::vector<Uint> prefixes(nSeqs);
  for(Uint i=0, prefix=0; i<nSeqs; ++i) {
    prefixes[i] = prefix;
    prefix += episodes[i]->ndata();
  }

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(Uint i=0; i<nSeqs; ++i)
  {
    auto & EP = * episodes[i].get();
    const Uint ndata = EP.ndata();
    const auto probs_i = probs.data() + prefixes[i];
    for(Uint j=0; j<ndata; ++j)
    {
      const float deltasq = EP.SquaredError(j);
      assert(deltasq > 0);
      // do sqrt(delta^2)^alpha with alpha = 0.5
      const float P = std::sqrt(std::sqrt(deltasq+EPS));
      minP = std::min(minP, P);
      maxP = std::max(maxP, P);
      EP.priorityImpW[j] = P;
      probs_i[j] = P;
    }
  }
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
void Sample_impSeq::prepare()
{
  const float EPS = std::numeric_limits<float>::epsilon();
  const long nSeqs = nEpisodes();
  std::vector<float> probs = std::vector<float>(nSeqs, 1);

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(long i=0; i<nSeqs; ++i)
  {
    auto & EP = * episodes[i].get();
    const Uint ndata = EP.ndata();
    //sampling P is episode's RMSE to the power beta=.5 times length of episode
    const float P = std::sqrt(std::sqrt(EP.avgSquaredErr + EPS)) * ndata;
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
      return episodes[a]->ndata() > episodes[b]->ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = episodes[seq[i]]->ndata() - 1;
  }
  else
  {
    std::uniform_real_distribution<float> distT(0, 1);
    std::vector<std::pair<Uint, Uint>> S (nBatch);

    std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
    while(it not_eq S.end())
    {
      std::generate(it, S.end(), [&] () {
         const Uint _s = distObs(gens[0]);
         const Uint _t = distT(gens[0]) * episodes[_s]->ndata();
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

  if(ret == nullptr) die("Setting dataSamplingAlgo not recognized.");
  return ret;
}

}
