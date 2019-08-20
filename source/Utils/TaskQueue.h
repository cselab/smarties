//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_TaskQueue_h
#define smarties_TaskQueue_h

#include <functional>
#include <utility>
#include <vector>

namespace smarties
{

class TaskQueue
{
  // some tasks may require to know whether data acquisition is globally locked
  const std::function<bool()> lockDataAcquisition;

  std::vector< std::function<void()> > tasks;

public:
  void add (std::function<void()> && func)
  {
    tasks.emplace_back(std::move(func));
  }

  void run()
  {
    // go through task list once and execute all that are ready:
    for(size_t i=0; i<tasks.size(); ++i) tasks[i]();
  }

  bool dataAcquisitionIsLocked() const
  {
    return lockDataAcquisition();
  }

  TaskQueue(const std::function<bool()> lock) : lockDataAcquisition(lock) {}
};

} // end namespace smarties
#endif // smarties_Settings_h
