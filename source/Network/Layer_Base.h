//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Layers.h"

class BaseLayer: public Layer
{
  const Uint nInputs, nNeurons, bRecurrent, nOut_simd;
  const Function* const func;
  vector<nnReal> initVals;

 public:
  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nWeight.push_back(nOut_simd * (bRecurrent? nInputs + nNeurons : nInputs));
    nBiases.push_back(nNeurons);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(nNeurons);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<Real> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals.resize(size, 0);
    std::copy(initVals.begin(), initVals.end(), initVals.begin());
  }
  ~BaseLayer() {
    _dispose_object(func);
  }

  BaseLayer(Uint _ID, Uint _nInputs, Uint _nNeurons, string funcType, bool bRnn,
    bool bOut, Uint iLink) : Layer(_ID, _nNeurons, bOut, false, iLink),
    nInputs(_nInputs), nNeurons(_nNeurons), bRecurrent(bRnn),
    nOut_simd(roundUpSimd(_nNeurons)), func(makeFunction(funcType)) {
      spanCompInpGrads = _nInputs;
    }

  string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") "<<func->name()
     <<string(bOutput? " output ":" ")
     <<string(bRecurrent? "Recurrent-":"")
     <<"InnerProduct Layer of size:"<<nNeurons
     <<" linked to Layer:"<<ID-link
     <<" of size:"<<nInputs<<"\n";
    return o.str();
  }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    nnReal* const suminp = curr->X(ID); //array that contains W * Y_{-1} + B
    assert(para->NB(ID) == nNeurons);
    memcpy(suminp, para->B(ID), nNeurons*sizeof(nnReal));
    {
      const nnReal* const inputs = curr->Y(ID-link);
      const nnReal* const weight = para->W(ID);
      for (Uint i = 0; i < nInputs; i++)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; o++)
          suminp[o] += inputs[i] * W[o];
      }
    }
    if(bRecurrent && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
      for (Uint i = 0; i < nNeurons; i++)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; o++)
          suminp[o] += inputs[i] * W[o];
      }
    }
    func->eval(suminp, curr->Y(ID), nNeurons);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    {
      nnReal* const deltas = curr->E(ID);
      const nnReal* const suminp = curr->X(ID);
      const nnReal* const outval = curr->Y(ID);
      for(Uint o=0; o<nNeurons; o++)
        deltas[o] *= func->evalDiff(suminp[o], outval[o]);
    }

    Layer::backward(nInputs, nNeurons, nOut_simd, bRecurrent? nNeurons : 0,
            prev, curr, next, grad, para);
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * func->initFactor(nInputs, nNeurons);
    uniform_real_distribution<nnReal> dis(-init, init);
    {
      nnReal* const biases = para->B(ID);
      for(Uint o=0; o<nNeurons; o++)
        if(initVals.size() != nNeurons) biases[o] = dis(*gen);
        else biases[o] = func->inverse(initVals[o]);
    }
    {
      nnReal* const weight = para->W(ID);
      for(Uint i=0; i<nInputs;  i++) for(Uint o=0; o<nNeurons; o++)
        weight[o +nOut_simd*i] = dis(*gen);
      //if(std::fabs(fac-1)<nnEPS) orthogonalize(para, gen, init);
    }
    if(bRecurrent)
    {
      nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
      for(Uint i=0; i<nNeurons;  i++) for(Uint o=0; o<nNeurons; o++)
        weight[o +nOut_simd*i] = dis(*gen);
    }
  }

  //void transpose(const Parameters*const para) const override
  //{
    //const nnReal* const W   = para->W(ID);
    //      nnReal* const W_T = para->W_T(ID);
    //for(Uint i=0; i<nInputs;  i++)
    //  for(Uint o=0; o<nNeurons; o++) W_T[nInp_simd*o + i] = W[nOut_simd*i + o];
  //}

  void orthogonalize(const Parameters*const para, mt19937*const gen, const Real initFac) const
  {
    nnReal* const weight = para->W(ID);
    for(Uint i=1; i<nNeurons; i++)
    {
      nnReal v_d_v_pre = 0, v_d_v_post = 0;
      for(Uint k=0; k<nInputs; k++)
        v_d_v_pre += weight[i +nOut_simd*k] * weight[i +nOut_simd*k];
      if(v_d_v_pre<nnEPS) {die("init error");}

      for(Uint j=0; j<i; j++) {
        nnReal u_d_u = 0.0, v_d_u = 0.0;
        for(Uint k=0; k<nInputs; k++) {
          u_d_u += weight[j +nOut_simd*k] * weight[j +nOut_simd*k];
          v_d_u += weight[j +nOut_simd*k] * weight[i +nOut_simd*k];
        }
        if(u_d_u<nnEPS) {die("init error");}
        for(Uint k=0; k<nInputs; k++)
          weight[i +nOut_simd*k] -= v_d_u/u_d_u* weight[j +nOut_simd*k];
      }

      for(Uint k=0; k<nInputs; k++)
        v_d_v_post += weight[i +nOut_simd*k] * weight[i +nOut_simd*k];

      if(std::sqrt(v_d_v_post)<nnEPS) {
        uniform_real_distribution<nnReal> dis(-initFac, initFac);
        for(Uint k=0; k<nInputs; k++) weight[i +nOut_simd*k] = dis(*gen);
      } else {
        const nnReal fac = std::sqrt(v_d_v_pre)/std::sqrt(v_d_v_post);
        for(Uint k=0; k<nInputs; k++) weight[i +nOut_simd*k] *= fac;
      }
    }
  }
};
