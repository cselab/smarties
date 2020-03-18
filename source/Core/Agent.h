//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Dmitry Alexeev.
//

#ifndef smarties_Agent_h
#define smarties_Agent_h

#include "StateAction.h"
#include "../Utils/Bund.h"
#include <cstring> // memcpy

#include <atomic>
#include <random>
#define OUTBUFFSIZE 65536

namespace smarties
{

enum episodeStatus {INIT = 0, CONT, TERM, TRNC, FAIL};
enum learnerStatus {WORK = 0, KILL};

inline int status2int(const episodeStatus status) {
  if(status == INIT) return 0;
  if(status == CONT) return 1;
  if(status == TERM) return 2;
  if(status == TRNC) return 3;
  if(status == FAIL) return 4;
  die("unreachable"); return 0;
}

struct Agent
{
  const unsigned ID;
  const unsigned workerID;
  const unsigned localID;

  episodeStatus agentStatus = INIT;
  unsigned timeStepInEpisode = 0;

  learnerStatus learnStatus = WORK;
  unsigned learnerTimeStepID = 0;
  unsigned learnerGradStepID = 0;
  double learnerAvgCumulativeReward = 0;

  bool trackEpisodes = true;

  MDPdescriptor& MDP;
  const StateInfo  sInfo = StateInfo(MDP);
  const ActionInfo aInfo = ActionInfo(MDP);

  std::vector<double> sOld = std::vector<double>(MDP.dimState, 0); // previous state
  std::vector<double> state = std::vector<double>(MDP.dimState, 0);  // current state
  std::vector<double> action = std::vector<double>(MDP.dimAction, 0);
  double reward; // current reward
  double cumulativeRewards = 0;

  std::mt19937 generator;
  std::normal_distribution<Real> distribution;
  std::uniform_real_distribution<Real> safety;

  Agent(Uint _ID, Uint workID, Uint _localID, MDPdescriptor& _MDP) :
    ID(_ID), workerID(workID), localID(_localID), MDP(_MDP) {}

  void reset()
  {
    agentStatus = INIT;
    timeStepInEpisode=0;
    cumulativeRewards=0;
    reward=0;
  }

  template<typename T>
  void update(const episodeStatus E, const std::vector<T>& S, const double R)
  {
    assert( S.size() == sInfo.dim() );
    agentStatus = E;

    if(E == FAIL) { // app crash, probably
      reset();
      return;
    }

    std::swap(sOld, state); //what is stored in state now is sold
    state = std::vector<double>(S.begin(), S.end());
    reward = R;

    if(E == INIT) {
      cumulativeRewards = 0;
      timeStepInEpisode = 0;
    } else {
      cumulativeRewards += R;
      ++timeStepInEpisode;
    }
  }

  void setAction(const Uint label) {
    action = aInfo.label2actionMessage<double>(label);
  }
  Uint getDiscreteAction() const {
    return aInfo.actionMessage2label(action);
  }

  void setAction(const Rvec& _act) {
    action = std::vector<double>(_act.begin(), _act.end());
  }
  template<typename T = double>
  std::vector<T> getAction() const {
    if(MDP.bDiscreteActions)
      return std::vector<double>(action.begin(), action.end());
    else return aInfo.learnerAction2envAction(action);
  }

  template<typename T = nnReal>
  std::vector<T> getObservedState() const {
    return sInfo.state2observed<T>(state);
  }
  template<typename T = nnReal>
  std::vector<T> getObservedOldState() const {
    return sInfo.state2observed<T>(sOld);
  }

  void packStateMsg(void * const buffer) const // put agent's state into buffer
  {
    assert(buffer not_eq nullptr);
    char * msgPos = (char*) buffer;
    memcpy(msgPos, &localID,           sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &agentStatus,       sizeof(episodeStatus));
    msgPos +=                          sizeof(episodeStatus) ;
    memcpy(msgPos, &timeStepInEpisode, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &reward,            sizeof(double));
    msgPos +=                          sizeof(double) ;
    assert(state.size() == MDP.dimState);
    if(state.size())
    memcpy(msgPos,  state.data(),      sizeof(double) * state.size());
    msgPos +=                          sizeof(double) * state.size() ;
  }

  void unpackStateMsg(const void * const buffer) // get state from buffer
  {
    assert(buffer not_eq nullptr);
    const char * msgPos = (const char*) buffer;
    unsigned testAgentID, testStepID;

    memcpy(&testAgentID,       msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    assert(testAgentID == localID);
    memcpy(&agentStatus,       msgPos, sizeof(episodeStatus));
    msgPos +=                          sizeof(episodeStatus) ;
    memcpy(&testStepID,        msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;

    // if timeStepInEpisode==testStepID then agent was told to pack and
    // unpack the same state. happens e.g. if learner == worker
    //if(timeStepInEpisode not_eq testStepID)
    {
      std::swap(sOld, state); //what is stored in state now is sold
      if(agentStatus == INIT) {
        cumulativeRewards = 0;
        timeStepInEpisode = 0;
      } else {
        cumulativeRewards += reward;
        ++timeStepInEpisode;
      }
    }
    assert(testStepID == timeStepInEpisode);

    memcpy(&reward,            msgPos, sizeof(double));
    msgPos +=                          sizeof(double) ;
    assert(state.size() == MDP.dimState);
    if(state.size())
    memcpy( state.data(),      msgPos, sizeof(double) * state.size());
    msgPos +=                          sizeof(double) * state.size() ;
  }

  void packActionMsg(void * const buffer) const
  {
    assert(buffer not_eq nullptr);
    char * msgPos = (char*) buffer;
    memcpy(msgPos, &localID,           sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &learnStatus,       sizeof(learnerStatus));
    msgPos +=                          sizeof(learnerStatus) ;
    memcpy(msgPos, &learnerTimeStepID, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(msgPos, &learnerGradStepID, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    assert(action.size() == MDP.dimAction);
    memcpy(msgPos,  action.data(),     sizeof(double) * action.size());
    msgPos +=                          sizeof(double) * action.size() ;
  }

  void unpackActionMsg(const void * const buffer)
  {
    assert(buffer not_eq nullptr);
    const char * msgPos = (const char*) buffer;
    unsigned testAgentID;
    memcpy(&testAgentID,       msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(&learnStatus,       msgPos, sizeof(learnerStatus));
    msgPos +=                          sizeof(learnerStatus) ;
    memcpy(&learnerTimeStepID, msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    memcpy(&learnerGradStepID, msgPos, sizeof(unsigned));
    msgPos +=                          sizeof(unsigned) ;
    assert(action.size() == MDP.dimAction);
    memcpy( action.data(),     msgPos, sizeof(double) * action.size());
    msgPos +=                          sizeof(double) * action.size() ;
    assert( testAgentID == localID );
  }

  static unsigned getMessageAgentID(const void *const buffer)
  {
    return * (unsigned*) buffer;
  }
  static episodeStatus& messageEpisodeStatus(char * buffer)
  {
    buffer += sizeof(unsigned);
    return * (episodeStatus *) buffer;
  }
  static learnerStatus& messageLearnerStatus(char * buffer)
  {
    buffer += sizeof(unsigned);
    return * (learnerStatus *) buffer;
  }
  static size_t computeStateMsgSize(const size_t sDim)
  {
   return 2*sizeof(unsigned) + sizeof(episodeStatus) + (sDim+1)*sizeof(double);
  }
  static size_t computeActionMsgSize(const size_t aDim)
  {
   return 3*sizeof(unsigned) +sizeof(learnerStatus) + aDim*sizeof(double);
  }

  // for dumping to state-action-reward-policy binary log (writeBuffer):
  mutable float buf[OUTBUFFSIZE];
  mutable std::atomic<Uint> buffCnter {0};

  void writeBuffer(const char* const logpath, const int rank) const
  {
    if(buffCnter == 0) return;
    char cpath[1024];
    snprintf(cpath, 1024, "%s/agent%03d_rank%02d_obs.raw", logpath, ID, rank);
    FILE * pFile = fopen (cpath, "ab");

    fwrite (buf, sizeof(float), buffCnter, pFile);
    fflush(pFile); fclose(pFile);
    buffCnter = 0;
  }

  void writeData(const char* const logpath, const int rank,
                 const Rvec& mu, const Uint globalTstep) const
  {
    // possible race conditions, avoided by the fact that each worker
    // (and therefore agent) can only be handled by one thread at the time
    // atomic op is to make sure that counter gets flushed to all threads
    const Uint writesize = 4 + sInfo.dim() + aInfo.dim() + mu.size();
    if(OUTBUFFSIZE<writesize) die("Increase compile-time OUTBUFFSIZE variable");
    assert( buffCnter % writesize == 0 );
    if(buffCnter+writesize > OUTBUFFSIZE) writeBuffer(logpath, rank);
    Uint ind = buffCnter;
    buf[ind++] = globalTstep + 0.1;
    buf[ind++] = status2int(agentStatus) + 0.1;
    buf[ind++] = timeStepInEpisode + 0.1;
    assert( state.size() == sInfo.dim());
    for (Uint i=0; i<state.size(); ++i) buf[ind++] = (float) state[i];
    assert(action.size() == aInfo.dim());
    for (Uint i=0; i<action.size(); ++i) buf[ind++] = (float) action[i];
    buf[ind++] = reward;
    for (Uint i=0; i<mu.size(); ++i) buf[ind++] = (float) mu[i];

    buffCnter += writesize;
    assert(buffCnter == ind);
  }

  void checkNanInf() const
  {
    #ifndef NDEBUG
      const auto assertValid = [] (const double val) {
        assert( not std::isnan(val) and not std::isinf(val) );
      };
      for (Uint j=0; j<action.size(); ++j) assertValid(action[j]);
      for (Uint j=0; j<state.size(); ++j) assertValid(state[j]);
      assertValid(reward);
    #endif
  }

  void initializeActionSampling(std::mt19937 & gen)
  {
    MDP.sharedNoiseVecTic = std::vector<Real>(aInfo.dim());
    MDP.sharedNoiseVecToc = std::vector<Real>(aInfo.dim());
    generator.seed( gen() );
    distribution = std::normal_distribution<Real>(0.0, 1.0);
    safety = std::uniform_real_distribution<Real>(-NORMDIST_MAX, NORMDIST_MAX);
    resetActionNoise();
  }

  Real sampleClippedGaussian()
  {
    Real samp = distribution(generator);
    if(samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) return safety(generator);
    else return samp;
  }

  void resetActionNoise()
  {
    // only one agent (0) needs to set a noise vector, and
    // if noise is non shared then this does not matter:
    if(localID > 0 or not MDP.bAgentsShareNoise) return;
    for(Uint i=0; i<aInfo.dim(); ++i)
      MDP.sharedNoiseVecTic[i] = sampleClippedGaussian();
    for(Uint i=0; i<aInfo.dim(); ++i)
      MDP.sharedNoiseVecToc[i] = sampleClippedGaussian();
  }

  Rvec sampleActionNoise()
  {
    if(MDP.bAgentsShareNoise)
    {
      // tic toc scheme to avoid race conditions:
      const bool bTic = timeStepInEpisode % 2;
      if(localID == 0) // agent 0 prepares next noise vector for the team:
      {
        auto& nxtNoise = bTic? MDP.sharedNoiseVecToc : MDP.sharedNoiseVecTic;
        for(Uint i=0; i<aInfo.dim(); ++i) nxtNoise[i] = sampleClippedGaussian();
      }
      return bTic? MDP.sharedNoiseVecTic : MDP.sharedNoiseVecToc;
    }
    else
    {
      Rvec sample(aInfo.dim());
      for(Uint i=0; i<aInfo.dim(); ++i) sample[i] = sampleClippedGaussian();
      return sample;
    }
  }
};

} // end namespace smarties
#undef OUTBUFFSIZE
#endif // smarties_Agent_h
