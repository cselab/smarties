//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../include/smarties.h"

using Communicator = smarties::Communicator;

void runTraining(Communicator*const comm, int argc, char**argv)
{
  for(int i=0; i<argc; i++) {printf("arg: %s\n", argv[i]); fflush(0);}
}

int main (int argc, char** argv)
{
  environment_callback_t callback = [] (Communicator*const sc,
                                            const MPI_Comm mc,
                                            int _argc, char** _argv) {
     for(int i=0; i<_argc; i++) {printf("arg: %s\n",_argv[i]); fflush(0);}
  };
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  //e.run(runTraining);
  return 0;
}
