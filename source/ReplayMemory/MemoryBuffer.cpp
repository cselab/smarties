//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"
#include <iterator>
#include <algorithm>

MemoryBuffer::MemoryBuffer(const Settings&S, const Environment*const E):
 settings(S), env(E), sampler(prepareSampler(S, this)) {
  Set.reserve(settings.maxTotObsNum);
}

void MemoryBuffer::save(const std::string base, const Uint nStep, const bool bBackup)
{
  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
  fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
  fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
  fwrite(&invstd_reward, sizeof(Real),             1, wFile);
  fflush(wFile); fclose(wFile);

  if(bBackup) {
    std::ostringstream S; S<<std::setw(9)<<std::setfill('0')<<nStep;
    wFile = fopen((base+"scaling_"+S.str()+".raw").c_str(), "wb");
    fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
    fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
    fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
    fwrite(&invstd_reward, sizeof(Real),             1, wFile);
    fflush(wFile); fclose(wFile);
  }
}

void MemoryBuffer::restart(const std::string base)
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

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  for(auto& old_traj: Set) _dispose_object(old_traj);

  Set.clear(); //clear trajectories used for learning
  nTransitions = 0;
  nSequences = 0;
}

void MemoryBuffer::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  sampler->sample(seq, obs);

  sampled = seq;
  std::sort(sampled.begin(), sampled.end());
  sampled.erase(std::unique(sampled.begin(), sampled.end()), sampled.end());

  for(Uint i=0; i<seq.size(); i++) Set[seq[i]]->setSampled(obs[i]);
}

void MemoryBuffer::removeSequence(const Uint ind)
{
  assert(readNSeq()>0);
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(nTransitions >= Set[ind]->ndata());
  assert(Set[ind] not_eq nullptr);
  nSequences--;
  needs_pass = true;
  nTransitions -= Set[ind]->ndata();
  std::swap(Set[ind], Set.back());
  _dispose_object(Set.back());
  Set.pop_back();
  assert(nSequences == (long) Set.size());
}
void MemoryBuffer::pushBackSequence(Sequence*const seq)
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert( readNSeq() == (long) Set.size() and seq not_eq nullptr);
  const auto ind = Set.size();
  Set.push_back(seq);
  Set[ind]->prefix = ind>0? Set[ind-1]->prefix +Set[ind-1]->ndata() : 0;
  nTransitions += seq->ndata();
  needs_pass = true;
  nSequences++;
  assert( readNSeq() == (long) Set.size());
}

void MemoryBuffer::initialize()
{
  // All sequences obtained before this point should share the same time stamp
  for(Uint i=0;i<Set.size();i++) Set[i]->ID = nSeenSequences.load();

  needs_pass = true;
  sampler->prepare(needs_pass);
}

MemoryBuffer::~MemoryBuffer()
{
  for (auto & trash : Set) _dispose_object( trash);
}

void MemoryBuffer::checkNData() {
  #ifndef NDEBUG
    Uint cntSamp = 0;
    for(Uint i=0; i<Set.size(); i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions);
    assert(nSequences==(long)Set.size());
  #endif
}

Sampling* MemoryBuffer::prepareSampler(const Settings&S, MemoryBuffer* const R)
{
  Sampling* ret = nullptr;

  if(S.dataSamplingAlgo == "uniform") ret = new Sample_uniform(S, R);

  if(S.dataSamplingAlgo == "impLen")  ret = new Sample_impLen(S, R);

  if(S.dataSamplingAlgo == "shuffle") {
    ret = new TSample_shuffle(S, R);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERrank") {
    ret = new TSample_impRank(S, R);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERerr") {
    ret = new TSample_impErr(S, R);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERseq") ret = new Sample_impSeq(S, R);

  assert(ret not_eq nullptr);
  return ret;
}
