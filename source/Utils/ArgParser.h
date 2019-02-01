//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Dmitry Alexeev.
//

#pragma once
#include <getopt.h>
#include <map>
#include <string>
#include <vector>

enum argumentTypes { NONE, INT, REAL, CHAR, STRING };

namespace ArgParser
{
struct OptionStruct
{
  char   shortOpt;
  std::string longOpt;
  argumentTypes  type;
  std::string description;
  void*  value;

  template <typename T>
  OptionStruct(
    char _shortOpt,
    std::string _longOpt,
    argumentTypes _type,
    std::string _description,
    T* _val, T _defVal
  ) : shortOpt(_shortOpt), longOpt(_longOpt), type(_type),
      description(_description), value((void*)_val)
  {
    *_val = _defVal;
  }

  OptionStruct() {};
  ~OptionStruct() {};

};

class Parser
{
private:
  const std::vector<OptionStruct> opts;
  std::map<char, OptionStruct> optsMap;
  const int nOpt = opts.size();
  struct option* const long_options = new option[nOpt + 1];
  std::string ctrlString = "";

public:
  ~Parser() {delete [] long_options;}
  Parser(const std::vector<OptionStruct> optionsMap);
  void parse(int argc, char * const * argv, bool verbose = false);
};
}
