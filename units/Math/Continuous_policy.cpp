#include <gtest/gtest.h>

#include "smarties/Core/StateAction.h"
#include "smarties/Math/Discrete_policy.h"
#include "smarties/Math/Continuous_policy.h"

#include <string.h> // memcpy
#include <algorithm>
#include <iostream>
#include <random>

#include <omp.h>
#include <mpi.h>

static constexpr double dbltol = std::numeric_limits<double>::epsilon();
static constexpr float  flttol = std::numeric_limits< float>::epsilon();


template<typename Policy_t>
inline void testPolicy(const smarties::ActionInfo & aI)
{
  std::random_device rdev;
  std::mt19937 gen(rdev());

  const int nA = Policy_t::compute_nA(aI);
  const int nPol = Policy_t::compute_nPol(aI);
  std::vector<double> piVec(nPol), muVec(nPol);
  // inds that map from beta vector to policy:
  // First nA component are either mean or option-probabilities and will have
  // size nA. Continuous policy has other nA components for stdev.
  std::normal_distribution<double> dist(0, 1);
  for(int i=0; i<nPol; ++i) muVec[i] = dist(gen);
  for(int i=0; i<nPol; ++i) piVec[i] = muVec[i] + 0.01*dist(gen);
  Policy_t PI(aI, piVec), MU(aI, muVec);
  typename Policy_t::Action_t act = MU.sample(gen);

  std::vector<double> network_gradient(2 * aI.dim(), 0);
  std::vector<double> dklGrad = PI.KLDivGradient(MU);
  std::vector<double> polGrad = PI.policyGradient(act);

  for(int i = 0; i<nPol; ++i)
  {
    auto out_1 = PI.netOutputs, out_2 = PI.netOutputs;
    out_1[i] -= flttol;
    out_2[i] += flttol;
    Policy_t p1(aI, out_1), p2(aI, out_2);

    auto p_1 = p1.evalLogProbability(act), d_1 = p1.KLDivergence(MU);
    auto p_2 = p2.evalLogProbability(act), d_2 = p2.KLDivergence(MU);

    auto dklFinDiff = (d_2-d_1)/(2*flttol);
    auto polFinDiff = (p_2-p_1)/(2*flttol);
    auto dklScale = std::max({std::fabs(dklGrad[i]), std::fabs(dklFinDiff), 1.0});
    auto polScale = std::max({std::fabs(polGrad[i]), std::fabs(polFinDiff), 1.0});
    //std::cout << polFinDiff << " " << polGrad[i] << std::endl;
    //std::cout << dklFinDiff << " " << dklGrad[i] << std::endl;
    ASSERT_LT(std::fabs(dklGrad[i] - dklFinDiff)/dklScale, flttol);
    ASSERT_LT(std::fabs(polGrad[i] - polFinDiff)/polScale, flttol);
  }
}

TEST (Math, Discrete_policy)
{
  smarties::MDPdescriptor MDP;
  MDP.dimState = 1;
  MDP.dimAction = 2;              //2 action components
  MDP.discreteActionValues = {2, 7}; //2 and 7 action options
  const auto sync = [](void* buffer, size_t size) {}; //no op
  MDP.synchronize(sync);
  smarties::ActionInfo aI(MDP);
  ASSERT_EQ(aI.dimDiscrete(), 14);

  // smarties assumes that the network output is always in linear space
  std::vector<double> network_output(aI.dimDiscrete(), 0);
  smarties::Discrete_policy pol(aI, network_output);
  ASSERT_GT(pol.getVector()[0], 0);
  ASSERT_GT(pol.getVector()[1], 0);
  testPolicy<smarties::Discrete_policy>(aI);
}

TEST (Math, Continuous_policy)
{
  smarties::MDPdescriptor MDP;
  MDP.dimState = 1;
  MDP.dimAction = 2;              //2 action components
  MDP.lowerActionValue = {1, -4};
  MDP.upperActionValue = {2, 8};
  // 1 component is a gaussian, other is a beta
  MDP.bActionSpaceBounded = {false, true};
  const auto sync = [](void* buffer, size_t size) {}; //no op
  MDP.synchronize(sync);
  smarties::ActionInfo aI(MDP);

  // smarties assumes that the network output is always in linear space
  std::vector<double> network_output(2 * aI.dim(), 0);
  smarties::Continuous_policy pol(aI, network_output);
  //zero input vector corresponds to 0 mean in gaussian, 0.5 in beta
  //(beta is always bound in 0 to 1, rescaling is done after)
  ASSERT_LT(std::fabs(pol.getMean()[0]), dbltol);
  ASSERT_LT(std::fabs(pol.getMean()[1]-0.5), dbltol);
  ASSERT_GT(pol.getVariance()[0], 0);
  ASSERT_GT(pol.getVariance()[1], 0);
  testPolicy<smarties::Continuous_policy>(aI);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
