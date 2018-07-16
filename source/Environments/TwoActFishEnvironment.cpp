//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "TwoActFishEnvironment.h"
//#define __Cubism3D

TwoActFishEnvironment::TwoActFishEnvironment(Settings& _settings) :
Environment(_settings), study(_settings.rType),
sight( _settings.senses    ==0 ||  _settings.senses==8),
rcast( _settings.senses    % 2), //if eq {1,  3,  5,  7}
lline((_settings.senses/2) % 2), //if eq {  2,3,    6,7}
press((_settings.senses/4) % 2 ||  _settings.senses==8), //if eq {      4,5,6,7}
goalDY((_settings.goalDY>1)? 1-_settings.goalDY : _settings.goalDY)
{
  printf("TwoActFishEnvironment.\n");

  //#ifdef __Cubism3D
    //mpi_ranks_per_env = 1;
    mpi_ranks_per_env = 2;
  //#else
    mpi_ranks_per_env = 0;
  //#endif
  //paramsfile="settings_32.txt";
  paramsfile="settings_64.txt";
  assert(settings.senses<=8);
}

void TwoActFishEnvironment::setDims()
{
  nAgentsPerRank = 1;
  aI.discrete = false;

  sI.inUse.clear();
  {
    // State: Horizontal distance from goal point...
    sI.inUse.push_back(sight);
    // ...vertical distance...
    sI.inUse.push_back(sight);
    // ...inclination of1the fish...
    sI.inUse.push_back(sight);
    // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
    sI.inUse.push_back(true);
    // ...last action (HAX!)
    sI.inUse.push_back(true);
    // ...second last action (HAX!)
    sI.inUse.push_back(true); //if l_line i have curvature info
  }
  {
    //New T period
    sI.inUse.push_back(true);

    //Phase Shift
    sI.inUse.push_back(true);

    // VxInst
    //sI.inUse.push_back(false);
    sI.inUse.push_back(true);

    // VyInst
    sI.inUse.push_back(true);

    // AvInst
    sI.inUse.push_back(true);
  }
  #if 0
    //Xabs 6
    sI.inUse.push_back(false);

    //Yabs 7
    sI.inUse.push_back(false);
  #endif
  {
    //Dist 6
    sI.inUse.push_back(false);

    //Quad 7
    sI.inUse.push_back(false);

    // VxAvg 8
    sI.inUse.push_back(false);

    // VyAvg 9
    sI.inUse.push_back(false);

    // AvAvg 10
    sI.inUse.push_back(false);

    //Pout 11
    sI.inUse.push_back(false);

    //defPower 12
    sI.inUse.push_back(false);

    // EffPDef 13
    sI.inUse.push_back(false);

    // PoutBnd 14
    sI.inUse.push_back(false);

    // defPowerBnd 15
    sI.inUse.push_back(false);

    // EffPDefBnd 16
    sI.inUse.push_back(false);

    // Pthrust 17
    sI.inUse.push_back(false);

    // Pdrag 18
    sI.inUse.push_back(false);

    // ToD 19
    sI.inUse.push_back(false);
  }

  const Uint nSens = 10;
  for(Uint i=0;i<nSens;i++) sI.inUse.push_back(press&&i<4); //(FPAbove)x10 [40]
  for(Uint i=0;i<nSens;i++) sI.inUse.push_back(press&&i<4); //(FVAbove)x10 [45]
  for(Uint i=0;i<nSens;i++) sI.inUse.push_back(press&&i<4); //(FPBelow)x10 [50]
  for(Uint i=0;i<nSens;i++) sI.inUse.push_back(press&&i<4); //(FVBelow)x10 [55]

  //for (Uint i=0; i<2*nSensors; i++) {
  //  // (FVBelow ) x 5 [55]
  //  sI.inUse.push_back(rcast);
  //}
  /*
  sI.values.push_back(-.50);
  sI.values.push_back(-.25);
  sI.values.push_back(0.00);
  sI.values.push_back(0.25);
  sI.values.push_back(0.50);
  */

  aI.dim = 2;
  aI.values.resize(2);
  //curavture
  aI.bounded.push_back(1);
  aI.values[0].push_back(-.75);
  aI.values[0].push_back(0.75);
  //period:
  aI.bounded.push_back(1);
  aI.values[1].push_back(-.5);
  aI.values[1].push_back(0.5);

  commonSetup();
}

#if 0
bool TwoActFishEnvironment::pickReward(const Agent& agent)
{
  if(info!=1)
  if (fabs(t_sO.vals[4] -t_sN.vals[5])>0.00001)
  _die("Mismatch two states [%s]==[%s]\n",t_sN._print().c_str(),t_a._print().c_str());

  bool new_sample(false);
  if (reward<-9.9) new_sample=true;
  if(new_sample) assert(info==2);

  if (study == 0) {
      #ifdef __Cubism3D
        reward = (t_sN.vals[18]-.3)/(.8-.6);
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
      #else
        reward = (t_sN.vals[18]-.3)/(1.-.3);
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
      #endif
  }
  else if (study == 1) {
      #ifdef __Cubism3D
        reward = (t_sN.vals[21]-.3)/(.6-.3);
      #else
        reward = (t_sN.vals[21]-.3)/(.6-.3);
      #endif

      if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
  }
  else if (study == 2) {
      reward =  1.-2*sqrt(fabs(t_sN.vals[1])); //-goalDY
      if (new_sample) reward = -2./(1.-gamma);
  }
  else if (study == 4) {
    const Real x=t_sN.vals[0], y=t_sN.vals[1];
    //Fish should stay 1.5 body lengths behind leader at y=0
    reward = 1 - std::sqrt((x-1.5)*(x-1.5) + y*y);
    if (new_sample) reward = -1./(1.-gamma);
  }
  else if (study == 5) {
    reward = (t_sN.vals[18]-.4)/.5;
    if (t_sN.vals[0] > 0.5) reward = std::min(0.,reward);
    if (new_sample) reward = -2./(1.-gamma);
  }
  else if (new_sample) reward = -10.;

  //gently push sim away from extreme curvature: not kosher
  if(std::fabs(t_a.vals[0])>0.74)
    reward = std::min((Real)0.,reward);
  if(std::fabs(t_a.vals[1])>0.49)
    reward = std::min((Real)0.,reward);

  return new_sample;
}
#endif
