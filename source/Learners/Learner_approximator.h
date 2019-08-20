//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Learner_approximator_h
#define smarties_Learner_approximator_h

#include "Learner.h"

namespace smarties
{

struct Approximator;

class Learner_approximator: public Learner
{
 protected:
  std::vector<Approximator*> networks;

  void initializeApproximators();
  bool createEncoder();

  virtual void Train(const MiniBatch&MB, const Uint wID,const Uint bID) const=0;
  void spawnTrainTasks();
  virtual void prepareGradient();
  virtual void applyGradient();

 public:
  Learner_approximator(MDPdescriptor&, Settings&, DistributionInfo&);
  virtual ~Learner_approximator() override;

  virtual void setupDataCollectionTasks(TaskQueue& tasks) override;

  virtual void getMetrics(std::ostringstream& buff) const override;
  virtual void getHeaders(std::ostringstream& buff) const override;
  virtual void save() override;
  virtual void restart() override;
};

}
#endif
