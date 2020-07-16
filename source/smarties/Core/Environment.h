//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Environment_h
#define smarties_Environment_h

#include "Agent.h"
#include <memory>

namespace smarties
{

struct Environment
{
  Uint nAgents, nAgentsPerEnvironment = 1;
  bool bAgentsHaveSeparateMDPdescriptors = false;
  //Uint nMPIranksPerEnvironment = 1;
  bool bFinalized = false;

  std::vector<std::unique_ptr<MDPdescriptor>> descriptors;
  std::vector<std::unique_ptr<Agent>> agents;
  std::vector<bool> bTrainFromAgentData;

  MDPdescriptor& getDescriptor(int agentID = 0)
  {
    if(not bAgentsHaveSeparateMDPdescriptors) agentID = 0;
    assert(descriptors.size() > (Uint) agentID);
    return * descriptors[agentID].get();
  }

  Environment()
  {
    descriptors.emplace_back( std::make_unique<MDPdescriptor>() );
    descriptors.back()->localID = descriptors.size() - 1;
  }

  void synchronizeEnvironments(
    const std::function<void(void*, size_t)>& sendRecvFunc,
    const Uint nCallingEnvironments = 1)
  {
    if(bFinalized) die("Cannot synchronize env description multiple times");
    bFinalized = true;

    sendRecvFunc(&nAgentsPerEnvironment, 1 * sizeof(Uint) );
    sendRecvFunc(&bAgentsHaveSeparateMDPdescriptors, 1 * sizeof(bool) );

    //sendRecvFunc(&nMPIranksPerEnvironment, 1 * sizeof(Uint) );
    //if(nMPIranksPerEnvironment <= 0) {
    //  warn("Overriding nMPIranksPerEnvironment -> 1");
    //  nMPIranksPerEnvironment = 1;
    //}

    bTrainFromAgentData.resize(nAgentsPerEnvironment, true);
    sendRecvVectorFunc(sendRecvFunc, bTrainFromAgentData);

    nAgents = nAgentsPerEnvironment * nCallingEnvironments;
    //assert(nCallingEnvironments>0);

    initDescriptors(bAgentsHaveSeparateMDPdescriptors);
    const Uint nDescriptors = descriptors.size();
    for(Uint i=0; i<nDescriptors; ++i) descriptors[i]->synchronize(sendRecvFunc);

    assert(agents.size() == 0);
    agents.clear();
    agents.reserve(nAgents);
    for(Uint i=0; i<nAgents; ++i)
    {
      // contiguous agents belong to same environment
      const Uint workerID = i / nAgentsPerEnvironment;
      const Uint localID  = i % nAgentsPerEnvironment;
      // agent with the same ID on different environment have the same MDP
      const Uint descriptorID = i % nDescriptors;
      MDPdescriptor& D = * descriptors[descriptorID].get();
      agents.emplace_back( std::make_unique<Agent>(i, workerID, localID, D) );
      agents[i]->trackEpisodes = bTrainFromAgentData[localID];
    }
  }

  void initDescriptors(const bool areDifferent)
  {
    bAgentsHaveSeparateMDPdescriptors = areDifferent;
    Uint nDescriptors = areDifferent? nAgentsPerEnvironment : 1;

    if(descriptors.size() > nDescriptors) die("conflicts in problem definition");

    descriptors.reserve(nDescriptors);
    for(Uint i=descriptors.size(); i<nDescriptors; ++i) {
      descriptors.emplace_back( // initialize new descriptor with old
        std::make_unique<MDPdescriptor>( getDescriptor(0) ) );
      descriptors.back()->localID = descriptors.size() - 1;
    }
  }

  #if 0
  // for a given environment, size of the IRL reward dictionary
  virtual Uint getNumberRewardParameters();

  // compute the reward given a certain state and param vector
  virtual Real getReward(const std::vector<memReal> s, const Rvec params);

  // compute the gradient of the reward
  virtual Rvec getRewardGrad(const std::vector<memReal> s, const Rvec params);
  #endif
};

} // end namespace smarties
#endif // smarties_Environment_h
