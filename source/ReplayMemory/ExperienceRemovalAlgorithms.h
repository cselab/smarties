#ifndef smarties_ExperienceRemovalAlgorithms_h
#define smarties_ExperienceRemovalAlgorithms_h

#include "Sequence.h"

namespace smarties
{

// episode with smallest average min(1, pi/mu):
struct MostOffPolicyEp
{
  Real avgOnPolicyR = 0;
  Uint countOnPolicyEps = 0;
  void updateAvgOnPolReward(const Sequence & EP, const int ep_ind)
  {
    if(EP.nFarPolicySteps() > 0) return;
    countOnPolicyEps ++;
    avgOnPolicyR += (EP.totR - avgOnPolicyR) / countOnPolicyEps;
  }

  int indUndr = -1;
  Real avgClipImpW = 9e9, mostOffR = 0;
  void updateMostFarUndrPol(const Sequence & EP, const int ep_ind)
  {
    const Real EP_avgClipImpW = EP.avgImpW;
    if(EP_avgClipImpW < avgClipImpW) {
      indUndr = ep_ind;
      avgClipImpW = EP_avgClipImpW;
      mostOffR = EP.totR;
    }
  }

  int indOver = -1;
  Real fracFarOverPol = -1, fracFarUndrPol = -1;
  void updateMostFarOverPol(const Sequence & EP, const int ep_ind)
  {
    const Real EP_fracFarOverPol = EP.nFarOverPolSteps / (Real) EP.ndata();
    const Real EP_fracFarUndrPol = EP.nFarUndrPolSteps / (Real) EP.ndata();
    if(EP_fracFarOverPol > fracFarOverPol) {
      indOver = ep_ind;
      fracFarOverPol = EP_fracFarOverPol;
    }
    if(EP_fracFarUndrPol > fracFarUndrPol)
      fracFarUndrPol = EP_fracFarUndrPol;
  }

  void compare(const Sequence & EP, const int ep_ind)
  {
    updateAvgOnPolReward(EP, ep_ind);
    updateMostFarUndrPol(EP, ep_ind);
    updateMostFarOverPol(EP, ep_ind);
  }

  void compare(const MostOffPolicyEp & EP)
  {
    const Real Nown = countOnPolicyEps, Nep = EP.countOnPolicyEps;
    const Real Wown = Nown / (Nown+Nep), Wep = Nep / (Nown+Nep);
    avgOnPolicyR = avgOnPolicyR * Wown + EP.avgOnPolicyR * Wep;
    countOnPolicyEps += EP.countOnPolicyEps;

    if(EP.avgClipImpW < avgClipImpW) {
      indUndr=EP.indUndr;
      avgClipImpW=EP.avgClipImpW;
      mostOffR=EP.mostOffR;
    }
    if(EP.fracFarOverPol > fracFarOverPol) {
      indOver=EP.indOver;
      fracFarOverPol=EP.fracFarOverPol;
    }
    if(EP.fracFarUndrPol > fracFarUndrPol)
      fracFarUndrPol=EP.fracFarUndrPol;
  }

  Sint operator()(const Real tolFarPol)
  {
    // If totR of most on policy EP is lower than totR of most off policy EP
    // then do not delete anything. Else delete most off-policy EP.
    //if ( fracFarOverPol > fracFarUndrPol ) return indOver;
    //printf("fracFarOverPol:%g fracFarUndrPol:%g avgOnPolicyR:%g mostOffR:%g\n",
    //fracFarOverPol, fracFarUndrPol, avgOnPolicyR, mostOffR);
    if ( fracFarOverPol > 2 * tolFarPol ) return indOver;
    else if (avgOnPolicyR > mostOffR) return indUndr;
    else return -1;
  }
};

// episode with highest fraction of far-policy steps as described in refer
struct MostFarPolicyEp
{
  int ind = -1;
  Real fracFarPol = -1;
  void compare(const Sequence & EP, const int ep_ind)
  {
    const Real EP_fracFarPol = EP.nFarPolicySteps()  / (Real) EP.ndata();
    if(EP_fracFarPol>fracFarPol) {
      ind = ep_ind;
      fracFarPol = EP_fracFarPol;
    }
  }

  void compare(const MostFarPolicyEp & EP)
  {
    if(EP.fracFarPol>fracFarPol) {
      ind = EP.ind;
      fracFarPol = EP.fracFarPol;
    }
  }

  Sint operator()() {
    return ind;
  }
};

// episode with highest average Kullback Leibler divergence
struct HighestAvgDklEp
{
  int ind = -1;
  Real averageDkl = -1;
  void compare(const Sequence & EP, const int ep_ind)
  {
    const Real EP_avgDkl = EP.sumKLDivergence / EP.ndata();
    if(EP_avgDkl > averageDkl) {
      ind = ep_ind;
      averageDkl = EP_avgDkl;
    }
  }

  void compare(const HighestAvgDklEp & EP)
  {
    if(EP.averageDkl>averageDkl) {
      ind = EP.ind;
      averageDkl = EP.averageDkl;
    }
  }

  Sint operator()() {
    return ind;
  }
};

// episode that was stored the most timesteps ago:
struct OldestDatasetEp
{
  int ind = -1;
  Sint timestamp = std::numeric_limits<Sint>::max();
  void compare(const Sequence & EP, const int ep_ind)
  {
    if(EP.ID < timestamp) {
      ind = ep_ind;
      timestamp = EP.ID;
    }
  }

  void compare(const OldestDatasetEp & EP)
  {
    if(EP.timestamp < timestamp) {
      ind = EP.ind;
      timestamp = EP.timestamp;
    }
  }

  Sint operator()() {
    return ind;
  }
};


FORGET MemoryProcessing::readERfilterAlgo(const std::string setting, const bool bReFER)
{
  const int world_rank = MPICommRank(MPI_COMM_WORLD);
  if(setting == "oldest") {
    if(world_rank == 0)
    printf("Experience Replay storage: First In First Out.\n");
    return OLDEST;
  }
  if(setting == "farpolfrac") {
    if(world_rank == 0)
    printf("Experience Replay storage: remove most 'far policy' episode.\n");
    return FARPOLFRAC;
  }
  if(setting == "maxkldiv") {
    if(world_rank == 0)
    printf("Experience Replay storage: remove highest average DKL episode.\n");
    return MAXKLDIV;
  }
  if(setting == "batchrl") {
    if(world_rank == 0)
    printf("Experience Replay storage: remove most 'off policy' episode if and only if policy is better.\n");
    return BATCHRL;
  }
  //if(setting == "minerror")   return MINERROR; miriad ways this can go wrong
  if(setting == "default") {
    if(bReFER) {
      if(world_rank == 0)
      printf("Experience Replay storage: remove most 'off policy' episode if and only if policy is better.\n");
      return BATCHRL;
    }
    else {
      if(world_rank == 0)
      printf("Experience Replay storage: First In First Out.\n");
      return OLDEST;
    }
  }
  die("ERoldSeqFilter not recognized");
  return OLDEST; // to silence warning
}

}
#endif // smarties_ExperienceRemovalAlgorithms_h
