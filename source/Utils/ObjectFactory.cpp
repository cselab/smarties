//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Dmitry Alexeev.
//

#include "ObjectFactory.h"
#include "../Environments/AllSystems.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <iostream>

using namespace std;

inline string ObjectFactory::_parse(string source, string pattern, bool req)
{
  size_t pos = source.find(((string)" ")+pattern);
  if (pos == string::npos) {
    if (req)
      _die("Parse factory file failed: required argument '%s' line '%s'\n",
          pattern.c_str(), source.c_str());
    else return "";
  }

  pos += pattern.length()+1;
  while (source[pos] == ' ') pos++;
  if (source[pos] != '=')
    _die("Parse factory file failed: argument '%s' line '%s'\n",
        pattern.c_str(), source.c_str());
        while (source[pos] == ' ') pos++;

  pos++;
  size_t stpos = pos;
  while (source[pos] != ' ' && pos < source.length()) pos++;

  return source.substr(stpos, pos - stpos);
}

inline int ObjectFactory::_parseInt(string source, string pattern, bool req)
{
  return atoi(_parse(source, pattern, req).c_str());
}

inline Real ObjectFactory::_parseReal(string source, string pattern, bool req)
{
  return atof(_parse(source, pattern, req).c_str());
}

Environment* ObjectFactory::createEnvironment()
{
  Environment* env = nullptr;
  string envStr = settings->environment;
  if(envStr=="TwoActFishEnvironment") env=new TwoActFishEnvironment(*settings);
  if(envStr=="AtariEnvironment")      env=new AtariEnvironment(*settings);
  else env = new Environment(*settings);
  if(env == nullptr) die("Env cannot be nullptr\n");
  return env;
}
