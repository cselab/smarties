//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "AtariEnvironment.h"
#include "../Network/Builder.h"

AtariEnvironment::AtariEnvironment(Settings& _sett): Environment(_sett)
{
  printf("AtariEnvironment.\n");
}

bool AtariEnvironment::predefinedNetwork(Builder & input_net) const
{
  assert(input_net.nInputs);
  // CNN is entirely templated for speed!
  input_net.addConv2d<LRelu, //nonlineariy
                  84, 84, 4,  // size of iunput x, y, c
                  8, 8, 32,   // size of kernel x, y, c
                  4, 4>();    // size of stride x, y

  input_net.addConv2d<LRelu, //nonlineariy
                  20, 20, 32, // size of iunput x, y, c
                  4, 4, 64,   // size of kernel x, y, c
                  2, 2>();    // size of stride x, y

  input_net.addConv2d<LRelu, //nonlineariy
                  9, 9, 64,   // size of iunput x, y, c
                  3, 3, 64,   // size of kernel x, y, c
                  1, 1>(true);    // size of stride x, y
  return true;
}
