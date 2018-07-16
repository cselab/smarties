//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
//#define __DBG_CNN
#include "Environment.h"

class AtariEnvironment: public Environment
{
public:
    AtariEnvironment(Settings & _settings);
    bool predefinedNetwork(Builder & input_net) const override;
};
