//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Diego Rossinelli.
//

#ifndef smarties_Profiler_h
#define smarties_Profiler_h

#include <string>
#include <chrono>
#include <unordered_map>

namespace smarties
{

class Timer
{
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _start, _end;

 public:
  Timer();
  void start();
  void stop();
  int64_t elapsed();
  int64_t elapsedAndReset();
};

struct Timings
{
  bool started;
  int iterations;
  int64_t total;
  Timer timer;

  Timings();
};

class Profiler
{
 public:
  enum Unit {s, ms, us};

 private:
  std::unordered_map<std::string, Timings>  timings;

  std::string ongoing;
  int numStarted;

  std::string __printStatAndReset(Unit unit, std::string prefix);

 public:
  Profiler();

  void start(std::string name);

  void stop();

  void stop_start(std::string name);

  double elapsed(std::string name, Unit unit = Unit::ms);

  std::string printStatAndReset(Unit unit = Unit::ms);

  void reset();
};

}
#endif
