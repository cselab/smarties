#include <gtest/gtest.h>

#include "smarties/Core/StateAction.h"

#include <string.h> // memcpy
#include <iostream>
#include <functional> // function

#include <omp.h>
#include <mpi.h>

TEST (Core, ActionInfoDiscrete)
{
  smarties::MDPdescriptor MDP;
  MDP.dimState = 1;
  MDP.dimAction = 2;              //2 action components
  MDP.discreteActionValues = {5, 3}; //5 and 3 action options
  const auto sync = [](void* buffer, size_t size) {}; //no op
  MDP.synchronize(sync);

  //for(Uint i=0; i<MDP.maxActionLabel; ++i)
  //  if(i != action2label(label2action(i)))
  //    _die("label %u does not match for action [%s]. returned %u",
  //      i, print(label2action(i)).c_str(), action2label(label2action(i)) );

  smarties::ActionInfo aI(MDP);
  // Action vector has N[0] * N[1] = 5 * 3 = 15 possible values.
  // Discrete algos will sample categorical distrib in [0, 15).
  // First action component is fast index, the second has stride 5.
  // Had there been a third comp, it would have stride N[0]*N[1].
  // Therefore action {2,1} has label 2 + 1 * 5 = 7
  // NOTE: because communication buffers assume floats, we add 0.1
  std::vector<double> action1 = {2.1, 1.1};
  ASSERT_EQ(aI.actionMessage2label(action1), 7);
  // Label 13 is action {13 % 5, 13 / 5}.
  std::cout << MDP.discreteActionShifts[0] << " "
            << MDP.discreteActionShifts[1] << std::endl;
  auto action2 = aI.label2actionMessage<double>(13);
  ASSERT_LT(std::fabs(action2[0] - 3.1), 0.5);
  ASSERT_LT(std::fabs(action2[1] - 2.1), 0.5);
  // It becomes a simple backwards for loop for higher dims.
}

TEST (Core, MDPsynchronize)
{
  smarties::MDPdescriptor MDPsrc;
  MDPsrc.dimState = 1;
  MDPsrc.dimAction = 1;
  smarties::MDPdescriptor MDPdst;

  void * exchange_buf = nullptr;
  size_t exchange_size = 0;

  omp_set_dynamic(0);
  omp_set_num_threads(2);

  // function called to make MDPsrc data available to dest
  const std::function<void(void*, size_t)> sendBuffer =
    [&](void* buffer, size_t size) {
    exchange_buf = buffer;
    exchange_size = size;
    #pragma omp barrier // dst begins copy
    #pragma omp barrier // dst ends copy
  };

  // function called by MDPdst to obtain MDPsrc data
  const std::function<void(void*, size_t)> recvBuffer =
    [&](void* buffer, size_t size) {
    #pragma omp barrier // src writes exchange
    ASSERT_EQ(exchange_size, size);
    memcpy(buffer, exchange_buf, size);
    exchange_buf = nullptr;
    exchange_size = 0;
    #pragma omp barrier // src may write again
  };

  #pragma omp parallel sections
  {
    #pragma omp section
    MDPsrc.synchronize(sendBuffer);
    #pragma omp section
    MDPdst.synchronize(recvBuffer);
  }

  ASSERT_EQ(MDPsrc.dimState,  MDPdst.dimState);
  ASSERT_EQ(MDPsrc.dimAction, MDPdst.dimAction);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
