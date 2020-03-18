//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../ReplayMemory/Collector.h"
#include "Learner_pytorch.h"
#include <chrono>

#ifdef PY11_PYTORCH

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;

// PYBIND11_MAKE_OPAQUE(smarties::Uint);
// PYBIND11_MAKE_OPAQUE(std::vector<smarties::Uint>);

PYBIND11_MAKE_OPAQUE(smarties::NNvec);
PYBIND11_MAKE_OPAQUE(std::vector< smarties::NNvec >);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::NNvec > >);
PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::Rvec* > >);
// PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::Real > >);
// PYBIND11_MAKE_OPAQUE(std::vector< std::vector< smarties::nnReal > >);  // <--- Do not convert to lists.
PYBIND11_MAKE_OPAQUE(std::vector<const smarties::MiniBatch*>);

namespace smarties
{

PYBIND11_EMBEDDED_MODULE(pybind11_embed, m) {
    py::class_<MiniBatch>(m, "MiniBatch")
        // .def(py::init<>())
        .def_readonly("begTimeStep", &MiniBatch::begTimeStep)
        .def_readonly("endTimeStep", &MiniBatch::endTimeStep)
        .def_readonly("sampledTimeStep", &MiniBatch::sampledTimeStep)
        .def_readwrite("S", &MiniBatch::S)
//        .def_readwrite("A", &MiniBatch::A)
//        .def_readwrite("MU", &MiniBatch::MU)
        .def_readwrite("R", &MiniBatch::R)
        .def_readwrite("PERW", &MiniBatch::PERW);

    // py::bind_vector<std::vector<Uint>>(m, "VectorUint");

    py::bind_vector<NNvec>(m, "NNvec");
    py::bind_vector<std::vector<NNvec>>(m, "VectorNNvec");
    py::bind_vector<std::vector<std::vector<NNvec>>>(m, "VectorVectorNNvec");
    // py::bind_vector<std::vector<std::vector<Real>>>(m, "VectorVectorReal");
    // py::bind_vector<std::vector< std::vector< nnReal > > >(m, "VectorVectornnReal");
    py::bind_vector<std::vector< std::vector< Rvec* > > >(m, "VectorVectorRvecPointer");
    py::bind_vector<std::vector<const MiniBatch*>>(m, "VectorMiniBatch");
}


py::scoped_interpreter guard{};

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner(MDP_, S_, D_)
{
  std::cout << "PYTORCH: STARTING NEW LEARNER." << std::endl;
  std::cout << "PYTORCH: Initializing pytorch scope..." << std::endl;

  py::module sys = py::module::import("sys");
  std::string path = "/home/pvlachas/smarties/source/Learners/Pytorch";
  sys.attr("path").attr("insert")(0,path);
  auto module = py::module::import("net_modules");
  auto Net = module.attr("NET");

  int input_dim = MDP_.dimStateObserved;
  // V-RACER outputing the value function, mean and std.
  int output_dim = 1 + aInfo.dim();
  int sigma_dim = aInfo.dim();

  // Outputing the mean action and the standard deviation of the action (possibly also the value function)
  std::cout << "PYTORCH: Output dimension (aInfo.dimPol())=" << output_dim << std::endl;

  auto net = Net(S_.nnType, S_.nnBPTTseq, input_dim, S_.nnLayerSizes, output_dim, sigma_dim);
  Nets.emplace_back(Net(S_.nnType, S_.nnBPTTseq, input_dim, S_.nnLayerSizes, output_dim, sigma_dim));
  Nets[0] = net;
  std::cout << "PYTORCH: NEW LEARNER STARTED." << std::endl;

  std::cout << "PYTORCH: PROPAGATING THROUGH THE LEARNER." << std::endl;
  auto torch = py::module::import("torch");
  int batch_size = 7;
  int time_steps = S_.nnBPTTseq+1;  
  auto input_ = torch.attr("randn")(batch_size, time_steps, input_dim);
  auto output = Nets[0].attr("forwardVector")(input_);
  std::cout << "PYTORCH: PROPAGATION WORKED!" << std::endl;

}

void Learner_pytorch::setupTasks(TaskQueue& tasks)
{
  std::cout << "PYTORCH: SETTING UP TASKS..." << std::endl;

  // If not training (e.g. evaluate policy)
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization
  auto stepInit = [&]()
  {
    // conditions to start the initialization task:
    if ( algoSubStepID >= 0 ) return; // we done with init
    if ( data->readNData() < nObsB4StartTraining ) return; // not enough data to init

    debugL("Initialize Learner");
    initializeLearner();
    algoSubStepID = 0;
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    // // debugL("Gather gradient estimates from each thread and Learner MPI rank");
    // // prepareGradient();
    // debugL("Search work to do in the Replay Memory");
    // processMemoryBuffer(); // find old eps, update avg quantities ...
    // debugL("Update Retrace est. for episodes sampled in prev. grad update");
    // updateRetraceEstimates();
    // debugL("Compute state/rewards stats from the replay memory");
    // finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
    logStats();
    
    algoSubStepID = 1;
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    // if ( networks[0]->ready2ApplyUpdate() == false ) return;

    // debugL("Apply SGD update after reduction of gradients");
    // applyGradient();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
  };
  tasks.add(stepComplete);

  std::cout << "PYTORCH: TASKS ALL SET UP..." << std::endl;
  std::cout << "PYTORCH: Data: " << data->readNSeq() << std::endl;
}



void Learner_pytorch::spawnTrainTasks()
{
  if(settings.bSampleEpisodes && data->readNSeq() < (long) settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  const Uint nThr = distrib.nThreads, CS =  batchSize / nThr;
  const MiniBatch MB = data->sampleMinibatch(batchSize, nGradSteps() );

  // IMPORTANT !
  py::module::import("pybind11_embed");

  std::vector<const MiniBatch*> vectorMiniBatch;
  vectorMiniBatch.push_back(&MB);
  std::reference_wrapper<std::vector<const MiniBatch*>> vectorMiniBatch_ref{vectorMiniBatch};
  auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref, "CmaxRet"_a=CmaxRet, "CinvRet"_a=CinvRet);

  // UPDATING RETRACE ESTIMATES
  for (Uint bID=0; bID<batchSize; ++bID) {
    Episode& S = MB.getEpisode(bID);
    const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();
    //Update Qret of eps' last state if sampled T-1. (and V(s_T) for truncated ep)
    if( S.isTruncated(t+1) ) {
      assert( t+1 == S.ndata() );
      std::cout << "! TRUNCATED DETECTED ! " << std::endl;

      // const Rvec nxt = NET.forward(bID, t+1);
      Fval fval = 0.0;
      // std::reference_wrapper<double> fval_ref{fval};
      std::cout << "BEFORE: FVAL IS " << fval << std::endl;
      auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref, "bID"_a=bID, "t"_a=t+1, "fval"_a=fval);
      auto output = Nets[0].attr("forwardBatchId")(locals);
      std::cout << "AFTER: FVAL IS " << fval << std::endl;

      // updateRetrace(S, t+1, 0, nxt[VsID], 0);
    }
  }
  auto output = Nets[0].attr("trainOnBatch")(locals);
}


void Learner_pytorch::select(Agent& agent)
{
  // std::cout << "PYTORCH: AGENT SELECTING ACTION!" << std::endl;

  data_get->add_state(agent);
  Episode& EP = data_get->get(agent.ID);

  const MiniBatch MB = data->agentToMinibatch(EP);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    // IMPORTANT !
    py::module::import("pybind11_embed");

    // Initializing action to be taken with zeros
    Rvec action = Rvec(aInfo.dim(), 0);
    // Initializing mu to be taken with zeros
    Rvec mu = Rvec(aInfo.dimPol(), 0);

    std::vector<const MiniBatch*> vectorMiniBatch;
    vectorMiniBatch.push_back(&MB);
    std::reference_wrapper<std::vector<const MiniBatch*>> vectorMiniBatch_ref{vectorMiniBatch};
    std::reference_wrapper<Rvec> action_ref{action};
    std::reference_wrapper<Rvec> mu_ref{mu};

    auto locals = py::dict("vectorMiniBatch"_a=vectorMiniBatch_ref, "action"_a=action_ref, "mu"_a=mu_ref);
    auto output = Nets[0].attr("selectAction")(locals);

    agent.act(action);
    data_get->add_action(agent, mu);

  } else {
    data_get->terminate_seq(agent);
  }

}

#else // not defined PY11_PYTORCH

namespace pybind11 {
struct object {};
}

namespace smarties
{

void Learner_pytorch::select(Agent& agent) {}

void Learner_pytorch::spawnTrainTasks() {}

void Learner_pytorch::setupTasks(TaskQueue& tasks) {}

Learner_pytorch::Learner_pytorch(MDPdescriptor& MDP_, Settings& S_,
  DistributionInfo& D_) : Learner(MDP_, S_, D_)
{}

#endif

Learner_pytorch::~Learner_pytorch()
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
  Learner::restart();
}

void Learner_pytorch::save()
{
  Learner::save();
}

}
