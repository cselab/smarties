//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Dmitry Alexeev.
//

#pragma once

#include <unistd.h>
#include <ctype.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#ifndef __APPLE__
inline long int mach_absolute_time()
{
  struct timeval clock;
  gettimeofday(&clock, NULL);
  /*
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);*/
  return (long int) (clock.tv_usec*1e6);
}
#else
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif

class Timer
{
private:
  long int _start, _end;

public:

  Timer()
{
    _start = 0;
    _end = 0;
}

  void start()
  {
    _start = mach_absolute_time();
    _end = 0;
  }

  void stop()
  {
    _end = mach_absolute_time();
  }

  long int elapsed()
  {
    if (_end == 0) _end = mach_absolute_time();
    return _end - _start;
  }

  long int elapsedAndReset()
  {
    if (_end == 0) _end = mach_absolute_time();
    long int t = _end - _start;
    _start = _end;
    _end = 0;
    return t;
  }

};
