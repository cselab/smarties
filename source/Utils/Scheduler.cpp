//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

Master::Master(Communicator_internal* const _c, const vector<Learner*> _l,
  Environment*const _e, Settings&_s): settings(_s),comm(_c),learners(_l),env(_e)
{
  profiler = new Profiler();
  for (Uint i=0; i<learners.size(); i++)  learners[i]->profiler = profiler;

  for(const auto& L : learners) // Figure out if I have on-pol learners
    bNeedSequentialTasks = bNeedSequentialTasks || L->bNeedSequentialTrain();

  if(nWorkers_own*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nWorkers nPerRank nAgents.");
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nWorkers_own; i++) comm->recvBuffer(i);
  profiler->stop_start("SLP");
  worker_replies.reserve(nWorkers_own);
}

void Master::run()
{
  { // gather initial data OR if not training evaluated restarted policy
    #pragma omp parallel num_threads(nThreads)
    {
      std::vector<int> shareWorkers;
      const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
      for(int i=1; i<=nWorkers_own; i++)
       if( thrID == (( ( (-i)%thrN ) +thrN ) %thrN) ) shareWorkers.push_back(i);

      #pragma omp critical
      if(shareWorkers.size()) worker_replies.push_back(
        std::thread( [&, shareWorkers] () { processWorker(shareWorkers); }));
    }
    while ( ! learnersInitialized() ) usleep(5);
  }
  if( not bTrain ) {
   for(size_t i=0; i<worker_replies.size(); i++) worker_replies[i].join();
   worker_replies.clear();
   return;
  }

  profiler->reset();
  for(const auto& L : learners) L->initializeLearner();

  while (true) // gradient step loop: one step per iteration
  {
    for(const auto& L : learners) L->spawnTrainTasks_par();

    if(bNeedSequentialTasks) {
      profiler->stop_start("SLP");
      // typically on-policy learning. Wait for all needed data:
      while ( ! learnersUnlockQueue() ) usleep(1);
      // and then perform on-policy update step(s):
      for(const auto& L : learners) L->spawnTrainTasks_seq();
    }

    for(const auto& L : learners) L->prepareGradient();

    if(not bNeedSequentialTasks) {
      profiler->stop_start("SLP");
      //for off-policy learners this is last possibility to wait for needed data
      while ( ! learnersUnlockQueue() ) usleep(1);
    }

    flushRewardBuffer();

    //This is the last possible time to finish the blocking mpi MPI_Allreduce
    // and finally perform the actual gradient step. Also, operations on memory
    // buffer that should be done after workers.join() are done here.
    for(const auto& L : learners) L->applyGradient();

    if( getMinStepId() >= totNumSteps ) {
      cout << "over!" << endl;
      return;
    }
  }
}

void Master::processWorker(const std::vector<int> workers)
{
  while(1)
  {
    if( not bTrain && getMinSeqId() >= totNumSteps) break;

    for( const int worker : workers )
    {
      assert(worker>0 && worker <= (int) nWorkers_own);
      int completed = comm->testBuffer(worker);

      // Learners lock workers queue if they have enough data to advance step
      while ( bTrain && completed && learnersLockQueue() ) {
        usleep(1);
        if( bExit.load() > 0 ) break;
      }

      if(completed) processAgent(worker);

      usleep(1);
    }
  }
}

void Master::processAgent(const int worker)
{
  //read from worker's buffer:
  vector<double> recv_state(sI.dim);
  int recv_agent  = -1; // id of agent inside environment
  int recv_status = -1; // initial/normal/termination/truncation of episode
  double reward   =  0;
  comm->unpackState(worker-1, recv_agent, recv_status, recv_state, reward);
  if (recv_status == FAIL_COMM) die("app crashed");

  const int agent = (worker-1) * nPerRank + recv_agent;
  Learner*const aAlgo = pickLearner(agent, recv_agent);


  agents[agent]->update(recv_status, recv_state, reward);
  //pick next action and ...do a bunch of other stuff with the data:
  aAlgo->select(*agents[agent]);

  debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]",
    agent, agents[agent]->Status, agents[agent]->sOld._print().c_str(),
    agents[agent]->s._print().c_str(), agents[agent]->r,
    agents[agent]->a._print().c_str());

  std::vector<double> actVec = agents[agent]->getAct();
  if(agents[agent]->Status >= TERM_COMM) actVec[0] = getMinStepId();
  debugS("Sent action to worker %d: [%s]", worker, print(actVec).c_str() );
  comm->sendBuffer(worker, actVec);

  if ( recv_status >= TERM_COMM )
    dumpCumulativeReward(recv_agent,worker,aAlgo->nStep(),aAlgo->tStepsTrain());

  comm->recvBuffer(worker);
}

Worker::Worker(Communicator_internal*const _c,Environment*const _e,Settings&_s)
: comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Worker::run() {
  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed

      if (comm->sendActionToApp() ) {
        die("Worker exiting");
        return;
      }
    }
    die("Simulation crash");
    //if we are training, then launch again, otherwise exit
    //if (!bTrain) return;
    comm->launch();
  }
}

void Master::flushRewardBuffer()
{
  for(int i=0; i<nPerRank; i++)
  {
    const Learner*const aAlgo = pickLearner(i, i);
    if( (iterNum % aAlgo->tPrint) not_eq 0 ) continue;

    ostringstream& agentBuf = rewardsBuffer[i];
    streampos pos = agentBuf.tellp(); // store current location
    agentBuf.seekp(0, ios_base::end); // go to end
    bool empty = agentBuf.tellp()==0; // check size == 0 ?
    agentBuf.seekp(pos);              // restore location
    if(empty) continue;               // else update rewards log
    char path[256];
    sprintf(path, "agent_%02d_rank%02d_cumulative_rewards.dat", i,learn_rank);
    ofstream outf(path, ios::app);
    outf << agentBuf.str();
    agentBuf.str(std::string());      // empty buffer
    outf.flush();
    outf.close();
  }
  iterNum++;
}

void Master::dumpCumulativeReward(const int agent, const int worker,
  const unsigned giter, const unsigned tstep) const
{
  if (giter == 0 && bTrain) return;

  const int ID = (worker-1) * nPerRank + agent;
  lock_guard<mutex> lock(dump_mutex);
  rewardsBuffer[agent]<<giter<<" "<<tstep<<" "<<worker<<" "
    <<agents[ID]->transitionID<<" "<<agents[ID]->cumulative_rewards<<endl;
  rewardsBuffer[agent].flush();
}

/*
Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
    Settings& _s):
    learner(_l), comm(_c), env(_e), agents(_e->agents), aI(_e->aI), sI(_e->sI),
    sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, &_s.generators[0]),
    aNew(_e->aI, &_s.generators[0]), status(_e->agents.size(),1)
{}

void Client::run()
{
  vector<double> state(env->sI.dim);
  int iAgent, agentStatus;
  double reward;

  while(true)
  {
    if (comm->recvStateFromApp()) break; //sim crashed

    prepareState(iAgent, agentStatus, reward);
    learner->select(iAgent, sNew, aNew, sOld, aOld, agentStatus, reward);

    debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
        iAgent, sOld._print().c_str(), sNew._print().c_str(),
        aOld._print().c_str(), reward, aNew._print().c_str());
    status[iAgent] = agentStatus;

    if(agentStatus != _AGENT_LASTCOMM) {
      prepareAction(iAgent);
      comm->sendActionToApp();
    } else {
      bool bDone = true; //did all agents reach terminal state?
      for (Uint i=0; i<status.size(); i++)
        bDone = bDone && status[i] == _AGENT_LASTCOMM;
      bDone = bDone || env->resetAll; //or does env end is any terminates?

      if(bDone) {
        comm->answerTerminateReq(-1);
        return;
      }
      else comm->answerTerminateReq(1);
    }
  }
}

void Client::prepareState(int& iAgent, int& istatus, Real& reward)
{
  Rvec recv_state(sNew.sInfo.dim);

  unpackState(comm->getDataState(), iAgent, istatus, recv_state, reward);
  assert(iAgent>=0 && iAgent<static_cast<int>(agents.size()));

  sNew.set(recv_state);
  //agent's s is stored in sOld
  agents[iAgent]->Status = istatus;
  agents[iAgent]->swapStates();
  agents[iAgent]->setState(sNew);
  agents[iAgent]->getOldState(sOld);
  agents[iAgent]->getAction(aOld);
  agents[iAgent]->r = reward;
}

void Client::prepareAction(const int iAgent)
{
  if(iAgent<0) die("Error in iAgent number in Client::prepareAction\n");
  assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
  agents[iAgent]->act(aNew);
  double* const buf = comm->getDataAction();
  for (Uint i=0; i<aI.dim; i++) buf[i] = aNew.vals[i];
}
*/
