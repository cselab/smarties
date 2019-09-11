//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Diego Rossinelli.
//

#include <algorithm>
#include "Profiler.h"
#include <vector>

#include <sstream>
#include <iomanip>
#include <cassert>

namespace smarties
{

Timer::Timer() {
  _start = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
  _end   = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
}

void Timer::start() {
  _start = std::chrono::high_resolution_clock::now();
  _end   = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
}

void Timer::stop() {
  _end = std::chrono::high_resolution_clock::now();
}

int64_t Timer::elapsed() {
  static constexpr auto none = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
  if (_end == none) _end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration <long int, std::nano>(_end - _start).count();
}

int64_t Timer::elapsedAndReset() {
  static constexpr auto none = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
  if (_end == none) _end = std::chrono::high_resolution_clock::now();

  int64_t t = std::chrono::duration <int64_t, std::nano>(_end - _start).count();

  _start = _end;
  _end = std::chrono::time_point<std::chrono::high_resolution_clock>::min();
  return t;
}

Timings::Timings() : started(false), iterations(0), total(0), timer() {}

std::string Profiler::__printStatAndReset(Unit unit, std::string prefix)
{
  double total = 0;
  unsigned longest = 0;
  std::ostringstream out;

  for (auto &tm : timings)
  {
    if (tm.second.started)
    {
      tm.second.started = false;
      tm.second.total += tm.second.timer.elapsed();
    }

    total += (double)tm.second.total;
    if (longest < tm.first.length())
      longest = tm.first.length();
  }

  double factor = 1.0;
  std::string suffix;
  switch (unit)
  {
    case Unit::s :
      factor = 1.0e-9;
      suffix = "s";
      break;

    case Unit::ms :
      factor = 1.0e-6;
      suffix = "ms";
      break;

    case Unit::us :
      factor = 1.0e-3;
      suffix = "us";
      break;
  }
  longest = std::max(longest, (unsigned)6);

  out<<prefix<<"Total time: "<<std::fixed<<std::setprecision(1)
                             <<total*factor<<" "<<suffix<<"\n";
  out<<prefix<<std::left<<"["<<std::setw(longest)<<"Kernel"<<"]    "<<std::setw(20)
     <<"Time, "+suffix<<std::setw(20)<<"Executions"<<std::setw(20)<<"Percentage";
  out<<"\n";
  std::vector<std::pair<std::string,Timings>> v(timings.begin(),timings.end());
  std::sort(v.begin(),v.end(),
    [] (auto& a, auto& b) { return a.second.total > b.second.total; });

  for (auto &tm : v)
  {
    if(tm.second.total <= 1e-2 * total) continue;
    out<<prefix<<"["<<std::setw(longest)<<tm.first<<"]    "
        <<std::fixed<<std::setprecision(3)<<std::setw(20)
                                 <<tm.second.total*factor / tm.second.iterations
               <<"x"<<std::setw(19)<<tm.second.iterations
        <<std::fixed<<std::setprecision(1)<<std::setw(20)
                                 <<tm.second.total/total * 100.0 << "\n";
  }

  timings.clear();

  return out.str();
}

void Profiler::start(std::string name)
{
  assert (ongoing.length() == 0);
  ongoing = name;
  auto& tm = timings[name];
  tm.started = true;
  tm.timer.start();
}

void Profiler::stop()
{
  assert(timings.find(ongoing) != timings.end());
  Timings &tm = timings[ongoing];
  if (tm.started)
  {
    tm.started = false;
    tm.total += tm.timer.elapsedAndReset();
    tm.iterations++;
  }
  ongoing = "";
}

void Profiler::stop_start(std::string name)
{
  if(ongoing not_eq "") stop();
  start(name);
}

double Profiler::elapsed(std::string name, Unit unit)
{
  double factor = 1.0;
  switch (unit)
  {
    case Unit::s :
      factor = 1.0e-9;
      break;

    case Unit::ms :
      factor = 1.0e-6;
      break;

    case Unit::us :
      factor = 1.0e-3;
      break;
  }

  if (timings.find(name) != timings.end())
  {
    Timings &tm = timings[name];
    return (factor * (double)tm.total) / tm.iterations;
  }
  return 0;
}

std::string Profiler::printStatAndReset(Unit unit)
{
  return __printStatAndReset(unit, "");
}

void Profiler::reset()
{
  timings.clear(); ongoing = ""; numStarted = 0;
}


Profiler::Profiler() : ongoing(""), numStarted(0) {}

}
