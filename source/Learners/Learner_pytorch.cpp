//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_pytorch.h"
#include <chrono>

namespace smarties
{

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner(MDP_, S_, D_)
{

}

Learner_pytorch::~Learner_pytorch() {
  // _dispose_object(input);
}

void Learner_pytorch::spawnTrainTasks()
{
}

void Learner_pytorch::getMetrics(std::ostringstream& buf) const
{
  Learner::getMetrics(buf);
}
void Learner_pytorch::getHeaders(std::ostringstream& buf) const
{
  Learner::getHeaders(buf);
}

void Learner_pytorch::restart()
{
  // if(settings.restart == "none") return;
  // if(!learn_rank) printf("Restarting from saved policy...\n");

  // for(auto & net : F) net->restart(settings.restart+"/"+learner_name);
  // input->restart(settings.restart+"/"+learner_name);

  // for(auto & net : F) net->save("restarted_"+learner_name, false);
  // input->save("restarted_"+learner_name, false);

  Learner::restart();

  // if(input->opt not_eq nullptr) input->opt->nStep = _nGradSteps;
  // for(auto & net : F) net->opt->nStep = _nGradSteps;
}

void Learner_pytorch::save()
{
  // const Uint currStep = nGradSteps()+1;
  // const Real freqSave = freqPrint * PRFL_DMPFRQ;
  // const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  // const bool bBackup = currStep % freqBackup == 0;
  // for(auto & net : F) net->save(learner_name, bBackup);
  // input->save(learner_name, bBackup);

  Learner::save();
}

}
