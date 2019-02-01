//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Dmitry Alexeev.
//

#include <cstdlib>
#include <map>
#include "ArgParser.h"
#include "Warnings.h"
#include "../Bund.h"

namespace ArgParser
{
Parser::Parser(const std::vector<OptionStruct> optionsMap) : opts(optionsMap)
{
  for (int i=0; i<nOpt; i++)
  {
    long_options[i].name = opts[i].longOpt.c_str();
    long_options[i].flag = NULL;
    long_options[i].val = opts[i].shortOpt;

    if (opts[i].type == NONE) long_options[i].has_arg = no_argument;
    else                      long_options[i].has_arg = required_argument;


    ctrlString += opts[i].shortOpt;
    if (opts[i].type != NONE) ctrlString += ':';

    if (optsMap.find(long_options[i].val) != optsMap.end())
      die("Duplicate short options in declaration. Correct Settings.h");
    else optsMap[long_options[i].val] = opts[i];

  }

  long_options[nOpt].has_arg = 0;
  long_options[nOpt].flag = NULL;
  long_options[nOpt].name = NULL;
  long_options[nOpt].val  = 0;
}

void Parser::parse(int argc, char * const * argv, bool verbose)
{
  int option_index = 0;
  int c = 0;

  while((c = getopt_long (argc, argv, ctrlString.c_str(), long_options, &option_index)) != -1)
  {
    if (c == 0) continue;
    if (verbose)
      if (optsMap.find(c) == optsMap.end())
      {
        printf("Available options:\n");

        for (int i=0; i<nOpt; i++)
        {
          const OptionStruct& myOpt = opts[i];
          if (myOpt.longOpt.length() > 4)
          {
            printf("-%c  or  --%s \t: %s\n", myOpt.shortOpt, myOpt.longOpt.c_str(), myOpt.description.c_str());
          }
          else
          {
            printf("-%c  or  --%s \t\t: %s\n", myOpt.shortOpt, myOpt.longOpt.c_str(), myOpt.description.c_str());
          }
        }

        die("Finishing program\n");
      }

    OptionStruct& myOpt = optsMap[c];

    switch (myOpt.type) {
    case NONE:
      *((bool*)myOpt.value) = true;
      break;

    case INT:
      *((int*)myOpt.value) = atoi(optarg);
      break;

    case REAL:
      *((Real*)myOpt.value) = atof(optarg);
      break;

    case CHAR:
      *((std::string*)myOpt.value) = optarg;
      break;

    case STRING:
      *((std::string*)myOpt.value) = optarg;
      break;
    }
  }

  if (verbose)
  {
    std::ofstream fout("settings.log", std::ios::app);
    if( not fout.is_open()) die("Save fail\n");

    for (int i=0; i<nOpt; i++) {
      const OptionStruct& myOpt = opts[i];
      fout << myOpt.description.c_str() <<": ";

      switch (myOpt.type) {
      case NONE:
        fout << ((*((bool*)myOpt.value)) ? "enabled" : "disabled");
        break;

      case INT:
        fout << (*((int*)myOpt.value));
        break;

      case REAL:
        fout << (*((Real*)myOpt.value));
        break;

      case CHAR:
        fout << (*((char*)myOpt.value));
        break;

      case STRING:
        fout << ((std::string*)myOpt.value)->c_str();
        break;
      }
      fout << std::endl;
    }
    fout.close();
  }
}
}
