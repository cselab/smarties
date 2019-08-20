//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Optimizer_h
#define smarties_Optimizer_h

#include "Layers/Parameters.h"
#include "../Settings.h"

namespace smarties
{

class Optimizer
{
protected:
  const DistributionInfo& distrib;
  const Settings & settings;
  const MPI_Comm learnersComm = MPICommDup(distrib.learners_train_comm);
  const Uint learn_size = MPICommSize(learnersComm);
  const Uint learn_rank = MPICommRank(learnersComm);
  const Uint populationSize = settings.ESpopSize;
  const Uint nThreads = distrib.nThreads;

  const std::shared_ptr<Parameters> weights;
  const std::vector<std::shared_ptr<Parameters>> sampled_weights =
                                       allocManyParams(weights, populationSize);
  const std::shared_ptr<Parameters> target_weights = weights->allocateEmptyAlike();

  Uint cntUpdateDelay = 0;
  std::mutex samples_mutex;

  std::vector<MPI_Request> weightsMPIrequests = std::vector<MPI_Request>(populationSize, MPI_REQUEST_NULL);

public:
  bool bAnnealLearnRate = true;
  const Real eta_init = settings.learnrate;
  nnReal eta = eta_init;
  const Uint batchSize = settings.batchSize;
  const Real lambda = settings.nnLambda;
  const Real epsAnneal = settings.epsAnneal;
  const Real tgtUpdateAlpha = settings.targetDelay;
  long unsigned nStep = 0;

  Optimizer(const Settings& S, const DistributionInfo& D,
            const std::shared_ptr<Parameters>& W);

  virtual ~Optimizer();
  virtual void save(const std::string fname, const bool bBackup) = 0;
  virtual int restart(const std::string fname) = 0;

  virtual void prepare_update(const Rvec&L) = 0;
  virtual void apply_update() = 0;

  virtual void getMetrics(std::ostringstream& buff) = 0;
  virtual void getHeaders(std::ostringstream&buff,const std::string nnName) = 0;
  virtual bool ready2UpdateWeights() = 0;

  const Parameters * getWeights(const Sint weightsIndex);
};

class AdamOptimizer : public Optimizer
{
protected:
  const Real beta_1, beta_2;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const std::vector<std::shared_ptr<Parameters>> gradients;
  const std::shared_ptr<Parameters> gradSum = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> _1stMom = weights->allocateEmptyAlike();
  const std::shared_ptr<Parameters> _2ndMom = weights->allocateEmptyAlike();
  std::vector<std::mt19937>& generators = distrib.generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;

public:

  AdamOptimizer(const Settings& S, const DistributionInfo& D,
                const std::shared_ptr<Parameters>& W,
                const std::vector<std::shared_ptr<Parameters>> & G,
                const Real B1=.9, const Real B2=.999);

  void prepare_update(const Rvec& L) override;
  bool ready2UpdateWeights() override
  {
    if(paramRequest == MPI_REQUEST_NULL) return true;
    int completed = 0;
    MPI(Test, &paramRequest, &completed, MPI_STATUS_IGNORE);
    return completed;
  }
  void apply_update() override;

  void save(const std::string fname, const bool bBackup) override;
  int restart(const std::string fname) override;
  void getMetrics(std::ostringstream& buff) override;
  void getHeaders(std::ostringstream& buff, const std::string nnName) override;
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
