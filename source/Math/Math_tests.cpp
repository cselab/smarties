//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Learners/AllLearners.h"
#include "../Math/Discrete_advantage.h"
#include "../Math/Gaussian_policy.h"


void Gaussian_policy::test(const Rvec& act, const Rvec& beta) const
{
  Rvec _grad(netOutputs.size(), 0);
   //const Rvec cntrolgrad = control_grad(a, -1);
   const Rvec div_klgrad = div_kl_grad(beta);
   const Rvec policygrad = policy_grad(act, 1);
   ofstream fout("mathtest.log", ios::app);
   for(Uint i = 0; i<2*nA; i++)
   {
     Rvec out_1 = netOutputs, out_2 = netOutputs;
     if(i>=nA && !start_prec) continue;
     const Uint index = i>=nA ? start_prec+i-nA : start_mean+i;
     out_1[index] -= nnEPS;
     out_2[index] += nnEPS;
     Gaussian_policy p1(vector<Uint>{start_mean, start_prec}, aInfo, out_1);
     Gaussian_policy p2(vector<Uint>{start_mean, start_prec}, aInfo, out_2);

    const Real p_1 = p1.evalLogProbability(act);
    const Real p_2 = p2.evalLogProbability(act);
    const Real d_1 = p1.kl_divergence(beta);
    const Real d_2 = p2.kl_divergence(beta);
    {
      finalize_grad(policygrad, _grad);
      const double diffVal = (p_2-p_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(p_2-p_1)/(2*nnEPS));
      fout<<"LogPol var grad "<<i<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
    }

    {
      finalize_grad(div_klgrad, _grad);
      const double diffVal = (d_2-d_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(d_2-d_1)/(2*nnEPS));
      fout<<"DivKL var grad "<<i<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
    }
   }
   fout.close();
}

void Discrete_policy::test(const Uint act, const Rvec& beta) const
{
  Rvec _grad(netOutputs.size());
  //const Rvec cntrolgrad = control_grad(-1);
  const Rvec div_klgrad = div_kl_grad(beta);
  const Rvec policygrad = policy_grad(act, 1);
  ofstream fout("mathtest.log", ios::app);
  //values_grad(act, 1, _grad);
  for(Uint i = 0; i<nA; i++)
  {
    Rvec out_1 = netOutputs, out_2 = netOutputs;
    const Uint index = start_prob+i;
    out_1[index] -= nnEPS;
    out_2[index] += nnEPS;
    Discrete_policy p1(vector<Uint>{start_prob},aInfo,out_1);
    Discrete_policy p2(vector<Uint>{start_prob},aInfo,out_2);
    //const Real A_1 = p1.computeAdvantage(act);
    //const Real A_2 = p2.computeAdvantage(act);
    const Real p_1 = p1.logProbability(act);
    const Real p_2 = p2.logProbability(act);
    const Real d_1 = p1.kl_divergence(beta);
    const Real d_2 = p2.kl_divergence(beta);

    finalize_grad(div_klgrad, _grad);
    {
      const double diffVal = (d_2-d_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(d_2-d_1)/(2*nnEPS));
      fout<<"DivKL var grad "<<i<<" "<<act<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
    }

    // finalize_grad(cntrolgrad, _grad);
    //if(fabs(_grad[index]-(A_2-A_1)/(2*nnEPS))>1e-7)
    // _die("Control grad %u %u: finite differences %g analytic %g error %g \n",
    //i,act,(A_2-A_1)/(2*nnEPS),_grad[index],fabs(_grad[index]-(A_2-A_1)/(2*nnEPS)));

    finalize_grad(policygrad, _grad);
    {
      const double diffVal = (p_2-p_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(p_2-p_1)/(2*nnEPS));
      fout<<"LogPol var grad "<<i<<" "<<act<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
    }
  }
  fout.close();
}
