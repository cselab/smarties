//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Dmitry Alexeev.
//

#pragma once
#include "../Agent.h"
#include "../Environments/Environment.h"

using namespace std;

class ObjectFactory
{
private:
  Settings * settings;
  inline string _parse(string source, string pattern, bool req = true);
  inline int    _parseInt(string source, string pattern, bool req = true);
  inline Real _parseReal(string source, string pattern, bool req = true);


public:
  ObjectFactory(Settings & _settings) : settings(&_settings) {}
  Environment* createEnvironment();
};
