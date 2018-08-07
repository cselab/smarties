//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Learners/AllLearners.h"

void Gaussian_policy::test(const Rvec& act, const Rvec& beta) const
{
  Rvec _grad(netOutputs.size());
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

    // Quadratic_advantage a1 =
    //  !a->start_mean ?
    //Quadratic_advantage(a->start_matrix,a->nA,a->nL,out_1,&p1)
    //  :
    //Quadratic_advantage(a->start_matrix,a->start_mean,a->nA,a->nL,out_1,&p1);

    //Quadratic_advantage a2 =
    //  !a->start_mean ?
    //Quadratic_advantage(a->start_matrix,a->nA,a->nL,out_2,&p2)
    //  :
    //Quadratic_advantage(a->start_matrix,a->start_mean,a->nA,a->nL,out_2,&p2);

    const Real p_1 = p1.logProbability(act);
    const Real p_2 = p2.logProbability(act);
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

    //#ifndef ACER_RELAX
    // const Real A_1 = a1.computeAdvantage(act);
    // const Real A_2 = a2.computeAdvantage(act);
    // finalize_grad_unb(cntrolgrad, _grad);
    //if(fabs(_grad[index]-(A_2-A_1)/(2*nnEPS))>1e-7)
    //_die("Control var grad %d: finite differences %g analytic %g error %g \n",
    //  i,(A_2-A_1)/(2*nnEPS),_grad[index],fabs(_grad[index]-(A_2-A_1)/(2*nnEPS)));
    //#endif

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

void Quadratic_advantage::test(const Rvec& act, mt19937*const gen) const
{
  Rvec _grad(netOutputs.size(), 0);
  grad(act, 1, _grad);
  ofstream fout("mathtest.log", ios::app);
  for(Uint i = 0; i<nL+nA; i++)
  {
    Rvec out_1 = netOutputs, out_2 = netOutputs;
    if(i>=nL && !start_mean) continue;
    const Uint index = i>=nL ? start_mean+i-nL : start_matrix+i;
    out_1[index] -= nnEPS;
    out_2[index] += nnEPS;

   Quadratic_advantage a1 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_1, policy);

   Quadratic_advantage a2 = Quadratic_advantage(vector<Uint>{start_matrix, start_mean}, aInfo, out_2, policy);

    const Real A_1 = a1.computeAdvantage(act);
    const Real A_2 = a2.computeAdvantage(act);
   {
     const double diffVal = (A_2-A_1)/(2*nnEPS);
     const double gradVal = _grad[index];
     const double errVal  = std::fabs(_grad[index]-(A_2-A_1)/(2*nnEPS));
     fout<<"Advantage grad "<<i<<" finite differences "
         <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
   }
  }
  fout.close();
}

void Diagonal_advantage::test(const Rvec& act, mt19937*const gen) const
{
  Rvec _grad(netOutputs.size());
  grad(act, 1, _grad);
  ofstream fout("mathtest.log", ios::app);
  for(Uint i = 0; i<4*nA; i++)
  {
    Rvec out_1 = netOutputs, out_2 = netOutputs;
    const Uint index = start_matrix+i;
    out_1[index] -= nnEPS;
    out_2[index] += nnEPS;

    Diagonal_advantage a1= Diagonal_advantage(vector<Uint>{start_matrix}, aInfo, out_1, policy);

    Diagonal_advantage a2= Diagonal_advantage(vector<Uint>{start_matrix}, aInfo, out_1, policy);

    const Real A_1 = a1.computeAdvantage(act);
    const Real A_2 = a2.computeAdvantage(act);
    {
      const double diffVal = (A_2-A_1)/(2*nnEPS);
      const double gradVal = _grad[index];
      const double errVal  = std::fabs(_grad[index]-(A_2-A_1)/(2*nnEPS));
      fout<<"Advantage grad "<<i<<" finite differences "
          <<diffVal<<" analytic "<<gradVal<<" error "<<errVal<<endl;
    }
  }
  fout.close();
}

void NAF::test()
{
  Rvec out(F[0]->nOutputs()), act(aInfo.dim);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  const int thrID = omp_get_thread_num();
  for(Uint i = 0; i<aInfo.dim; i++) act[i] = act_dis(generators[thrID]);
  for(Uint i = 0; i<F[0]->nOutputs(); i++) out[i] = out_dis(generators[thrID]);
  Quadratic_advantage A = prepare_advantage(out);
  A.test(act, &generators[thrID]);
}

/*
void CACER::test()
{
  Rvec hat(nOutputs), out(nOutputs), act(nA);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  for(Uint i = 0; i<nA; i++) act[i] = act_dis(*gen);
  Gaussian_policy pol_hat = prepare_policy(hat);
  Gaussian_policy pol_cur = prepare_policy(out);
  Quadratic_advantage adv = prepare_advantage(out, &pol_cur);
  pol_cur.test(act, &pol_hat); //,&adv
  adv.test(act);
}
*/

/*
void POAC::test()
{
  Rvec hat(nOutputs), out(nOutputs), act(nA);
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(-.5,.5);
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  for(Uint i = 0; i<nA; i++) act[i] = act_dis(*gen);
  Gaussian_policy pol_hat = prepare_policy(hat);
  Gaussian_policy pol_cur = prepare_policy(out);
  Advantage adv = prepare_advantage(out, &pol_cur);
  pol_cur.test(act, &pol_hat); //,&adv
  adv.test(act);
}
*/
/*
void DACER::test()
{
  uniform_real_distribution<Real> out_dis(-.5,.5);
  uniform_real_distribution<Real> act_dis(0, 1.);
  Rvec hat(nOutputs), out(nOutputs);
  const Uint act = act_dis(*gen)*nA;
  for(Uint i = 0; i<nOutputs; i++) out[i] = out_dis(*gen);
  for(Uint i = 0; i<nOutputs; i++) hat[i] = out_dis(*gen);
  Discrete_policy pol_hat = prepare_policy(hat);
  Discrete_policy pol_cur = prepare_policy(out);
  pol_cur.test(act, &pol_hat);
}
*/
