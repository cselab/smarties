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
: gens(G), RM(R), Set(RM->Set), bSampleSequences(bSeq) {}

long Sampling::nSequences() const { return RM->readNSeq(); }
long Sampling::nTransitions() const { return RM->readNData(); }
void Sampling::setMinMaxProb(const Real maxP, const Real minP) {
    RM->minPriorityImpW = minP;
    RM->maxPriorityImpW = maxP;
}

void Sampling::IDtoSeqStep(std::vector<Uint>& seq, std::vector<Uint>& obs,
  const std::vector<Uint>& ret, const Uint nSeqs)
{
  // go through each element of ret to find corresponding seq and obs
  const Uint Bsize = seq.size();
  for (Uint k = 0, cntO = 0, i = 0; k<nSeqs; ++k) {
    while(1) {
      assert(ret[i] >= cntO && i < Bsize);
      if(ret[i] < cntO + Set[k]->ndata()) { // is ret[i] in sequence k?
        obs[i] = ret[i] - cntO; //if ret[i]==cntO then obs 0 of k and so forth
        seq[i] = k;
        ++i; // next iteration remember first i-1 were already found
      }
      else break;
      if(i == Bsize) break; // then found all elements of sequence k
    }
    //assert(cntO == Set[k]->prefix);
    if(i == Bsize) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == nSeqs) die(" "); // at last iter we must have found all
  }
}



Sample_uniform::Sample_uniform(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_uniform::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  assert(seq.size() == obs.size());
  std::unique_lock<std::mutex> lock(RM->dataset_mutex, std::defer_lock);
  lock.lock();
  //std::lock_guard<std::mutex> lock(RM->dataset_mutex);
  const long nSeqs = nSequences(), nData = nTransitions(), nBatch = obs.size();
  lock.unlock();

  #ifndef NDEBUG
  assert(Set.size() == (size_t) nSeqs);
  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    assert(Set[i]->prefix == (Uint) locPrefix);
    locPrefix += Set[i]->ndata();
    if(i+1 == nSeqs) assert(locPrefix == nData);    
  }
  #endif

  if(bSampleSequences)
  {
    std::uniform_int_distribution<Uint> distSeq(0, nSeqs-1);
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return distSeq(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }

    const auto compare = [&](const Uint a, const Uint b) {
      return Set[a]->ndata() > Set[b]->ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (long i=0; i<nBatch; ++i) obs[i] = Set[seq[i]]->ndata() - 1;
  }
  else
  {
    std::uniform_int_distribution<Uint> distObs(0, nData-1);
    std::vector<Uint> ret(nBatch);
    std::vector<Uint>::iterator it = ret.begin();
    while(it not_eq ret.end())
    {
      std::generate(it, ret.end(), [&]() { return distObs(gens[0]); } );
      std::sort(ret.begin(), ret.end());
      it = std::unique (ret.begin(), ret.end());
    } // ret is now also sorted!

    IDtoSeqStep(seq, obs, ret, nSeqs);
  }
}
void Sample_uniform::prepare(std::atomic<bool>& needs_pass) {
  std::lock_guard<std::mutex> lock(RM->dataset_mutex);
  const long nSeqs = nSequences();
  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    Set[i]->prefix = locPrefix;
    locPrefix += Set[i]->ndata();
  }
  needs_pass = false;
}
bool Sample_uniform::requireImportanceWeights() { return false; }



Sample_impLen::Sample_impLen(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void Sample_impLen::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");
  const Uint nBatch = obs.size();

  if(bSampleSequences)
  {
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return dist(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }
    const auto compare = [&](const Uint a, const Uint b) {
      return Set[a]->ndata() > Set[b]->ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = Set[seq[i]]->ndata() - 1;
  }
  else
  {
    std::uniform_real_distribution<float> distT(0, 1);
    std::vector<std::pair<Uint, Uint>> S (nBatch);
    std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
    while(it not_eq S.end()) {
      std::generate(it, S.end(), [&] () {
          const Uint _s = dist(gens[0]), _t = distT(gens[0]) * Set[_s]->ndata();
          return std::pair<Uint, Uint> {_s, _t};
        }
      );
      std::sort( S.begin(), S.end() );
      it = std::unique( S.begin(), S.end() );
    }
    for (Uint i=0; i<nBatch; ++i) { seq[i] = S[i].first; obs[i] = S[i].second; }
  }
}
void Sample_impLen::prepare(std::atomic<bool>& needs_pass)
{
  if(not needs_pass) return;
  needs_pass = false;
  const Uint nSeqs = nSequences();
  std::vector<float> probs(nSeqs, 0);

  #pragma omp parallel for schedule(static)
  for (Uint i=0; i<nSeqs; ++i) probs[i] = Set[i]->ndata();

  dist = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}
bool Sample_impLen::requireImportanceWeights() { return false; }



TSample_shuffle::TSample_shuffle(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_shuffle::prepare(std::atomic<bool>& needs_pass)
{
  if(not needs_pass) return;
  needs_pass = false;

  const long nSeqs = nSequences(), nData = nTransitions();
  samples.resize(nData);

  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    Set[i]->prefix = locPrefix;
    locPrefix += Set[i]->ndata();
  }
  #pragma omp parallel for schedule(dynamic)
  for(long i = 0; i < nSeqs; ++i)
    for(Uint j=0, k=Set[i]->prefix; j<Set[i]->ndata(); ++j, ++k)
      samples[k] = std::pair<unsigned, unsigned>{i, j};

  const auto RNG = [&](const int max) {
    assert(max > 0);
    std::uniform_int_distribution<int> dist(0, max-1);
    return dist(gens[0]);
  };
  //__gnu_parallel::random_shuffle(samples.begin(), samples.end(), RNG);
  std::random_shuffle(samples.begin(), samples.end(), RNG);
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



static inline float approxRsqrt( const float number )
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

TSample_impRank::TSample_impRank(std::vector<std::mt19937>&G, MemoryBuffer*const R, bool bSeq): Sampling(G,R,bSeq) {}
void TSample_impRank::prepare(std::atomic<bool>& needs_pass)
{
  if( ( stepSinceISWeep++ >= 10 || needs_pass ) == false ) return;
  stepSinceISWeep = 0;
  needs_pass = false;
  // we need to collect all errors and rank them by magnitude
  // how do we manage to do most of the work in one pass?
  // 1) gather errors along with index
  // 2) sort them by decreasing error
  // 3) compute inv sqrt of all errors, same sweep also get minP
  //const float EPS = numeric_limits<float>::epsilon();
  using USI = unsigned short;
  using TupEST = std::tuple<float, USI, USI>;
  const long nSeqs = nSequences(), nData = nTransitions();
  if(nSeqs >= 65535) die("Too much data for data format");

  std::vector<TupEST> errors(nData);
  // 1)
  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    Set[i]->prefix = locPrefix;
    const auto err_i = errors.data() + locPrefix;
    locPrefix += Set[i]->ndata();
    for(Uint j=0; j<Set[i]->ndata(); ++j)
      err_i[j] = std::make_tuple(Set[i]->SquaredError[j], (USI)i, (USI)j );
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
    const float P = std::get<0>(errors[i])>0 ? approxRsqrt(i+1) : 1;
    const Uint seq = std::get<1>(errors[i]), t = std::get<2>(errors[i]);
    probs[Set[seq]->prefix + t] = P;
    Set[seq]->priorityImpW[t] = P;
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
  const long nSeqs = nSequences();
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
  const long nSeqs = nSequences(), nData = nTransitions();
  std::vector<float> probs = std::vector<float>(nData, 1);

  for(long i=0, locPrefix=0; i<nSeqs; ++i) {
    Set[i]->prefix = locPrefix;
    locPrefix += Set[i]->ndata();
  }

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(long i=0; i<nSeqs; ++i)
  {
    const Uint ndata = Set[i]->ndata();
    const auto probs_i = probs.data() + Set[i]->prefix;

    for(Uint j=0; j<ndata; ++j) {
      const float deltasq = (float) Set[i]->SquaredError[j];
      // do sqrt(delta^2)^alpha with alpha = 0.5
      const float P = deltasq>0.0f? std::sqrt(std::sqrt(deltasq+EPS)) : 0.0f;
      const float Pe = P + EPS;
      const float Qe = P>0.0f? P + EPS : 1.0e9f ; // avoid nans in impW
      minP = std::min(minP, Qe);
      maxP = std::max(maxP, Pe);
      Set[i]->priorityImpW[j] = P;
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
  const long nSeqs = nSequences();
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
  const long nSeqs = nSequences();
  std::vector<float> probs = std::vector<float>(nSeqs, 1);

  Fval maxMSE = 0;
  for(long i=0; i<nSeqs; ++i)
    maxMSE = std::max(maxMSE, Set[i]->MSE / Set[i]->ndata());

  float minP = 1e9, maxP = 0;
  #pragma omp parallel for schedule(dynamic) reduction(min:minP) reduction(max:maxP)
  for(long i=0; i<nSeqs; ++i)
  {
    const Uint ndata = Set[i]->ndata();
    float sumErrors = Set[i]->MSE;
    for(Uint j=0; j<ndata; ++j)
      if( std::fabs( Set[i]->SquaredError[j] ) <= 0 ) sumErrors += maxMSE;
    //sampling P is episode's RMSE to the power beta=.5 times length of episode
    const float P = std::sqrt( std::sqrt( (sumErrors + EPS)/ndata ) ) * ndata;
    std::fill(Set[i]->priorityImpW.begin(), Set[i]->priorityImpW.end(), P);
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

  if(bSampleSequences)
  {
    std::vector<Uint>::iterator it = seq.begin();
    while(it not_eq seq.end())
    {
      std::generate(it, seq.end(), [&]() { return distObs(gens[0]); } );
      std::sort( seq.begin(), seq.end() );
      it = std::unique( seq.begin(), seq.end() );
    }
    const auto compare = [&](const Uint a, const Uint b) {
      return Set[a]->ndata() > Set[b]->ndata();
    };
    std::sort(seq.begin(), seq.end(), compare);
    for (Uint i=0; i<nBatch; ++i) obs[i] = Set[seq[i]]->ndata() - 1;
  }
  else
  {
    std::uniform_real_distribution<float> distT(0, 1);
    std::vector<std::pair<Uint, Uint>> S (nBatch);

    std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
    while(it not_eq S.end()) {
      std::generate(it, S.end(), [&] () {
         const Uint _s = distObs(gens[0]), _t = distT(gens[0]) * Set[_s]->ndata();
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

}
