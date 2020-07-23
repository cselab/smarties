//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Learners/AlgoFactory.h"
#include "Discrete_advantage.h"
#include "Continuous_policy.h"
#include "Quadratic_advantage.h"
#include "Gaus_advantage.h"
#include <fstream>
#include <random>

namespace smarties
{
/*
template<typename Advantage_t, typename Policy_t, typename Action_t>
void testPolicyAdvantage(std::vector<Uint> polInds, std::vector<Uint> advInds,
  std::vector<Uint> netOuts, std::mt19937& gen, const ActionInfo& aI)
{
  const Uint nA = Policy_t::compute_nA(aI);
  const Uint nPol = Policy_t::compute_nPol(aI);
  Rvec mu(nPol), noise(nA);
  Rvec nnOut;
  std::normal_distribution<Real> dist(0, 1);
  for(Uint j=0; j < netOuts.size(); ++j)
      for(Uint i=0; i < netOuts[j]; ++i)
          nnOut.push_back(dist(gen));
  for(Uint i=0; i<nPol; ++i) mu[i] = dist(gen);
  Policy_t pol1(polInds, aI, nnOut);
  Policy_t pol2({0, nA}, aI, mu);
  Advantage_t adv(advInds, aI, nnOut, & pol1);
  Action_t act = pol1.sample(gen);
  pol1.test(act, pol2.getVector());
  adv.test(act, gen);
}

void Gaussian_advantage::test(const Rvec& act, std::mt19937& gen) const
{
  const Uint numNetOutputs = netOutputs.size();
  Rvec _grad(numNetOutputs, 0);
  grad(act, 1, _grad);
  std::ofstream fout("mathtest.log", std::ios::app);
  for(Uint i = 0; i<nL; ++i)
  {
    Rvec out_1 = netOutputs, out_2 = netOutputs;
    const Uint index = start_coefs+i;
    out_1[index] -= 0.0001; out_2[index] += 0.0001;

    Gaussian_advantage a1(std::vector<Uint>{start_coefs}, aInfo, out_1, policy);
    Gaussian_advantage a2(std::vector<Uint>{start_coefs}, aInfo, out_2, policy);
    const Real A_1 = a1.computeAdvantage(act), A_2 = a2.computeAdvantage(act);
    const Real fdiff =(A_2-A_1)/.0002, abserr = std::fabs(_grad[index]-fdiff);
    const Real scale = std::max(std::fabs(fdiff), std::fabs(_grad[index]));
    //if(abserr>1e-7 && abserr/scale>1e-4)
    {
      fout<<"Adv grad "<<i<<" finite differences "<<fdiff<<" analytic "
        <<_grad[index]<<" error "<<abserr<<" "<<abserr/scale<<"\n";
    }
  }
  fout.close();
}

void Quadratic_advantage::test(const Rvec& act, std::mt19937& gen) const
{
  Rvec _grad(netOutputs.size(), 0);
  grad(act, 1, _grad);
  std::ofstream fout("mathtest.log", std::ios::app);
  for(Uint i = 0; i<nL+nA; ++i)
  {
    Rvec out_1 = netOutputs, out_2 = netOutputs;
    if(i>=nL && start_mean == 0) continue;
    const Uint index = i>=nL ? start_mean+i-nL : start_matrix+i;
    out_1[index] -= nnEPS;
    out_2[index] += nnEPS;

    Quadratic_advantage a1 = Quadratic_advantage(std::vector<Uint>{start_matrix,
      start_mean}, aInfo, out_1, policy);

    Quadratic_advantage a2 = Quadratic_advantage(std::vector<Uint>{start_matrix,
      start_mean}, aInfo, out_2, policy);

    const Real A_1 = a1.computeAdvantage(act);
    const Real A_2 = a2.computeAdvantage(act);
    {
      const double diffVal = (A_2-A_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(A_2-A_1)/(2*nnEPS));
      fout<<"Advantage grad "<<i<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<"\n";
    }
  }
  fout.close();
}
*/
} // end namespace smarties
