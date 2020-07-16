//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_BaseLayer_h
#define smarties_BaseLayer_h

#include "Layers.h"

namespace smarties
{

class BaseLayer: public Layer
{
  const Uint nInputs, nNeurons, bRecurrent, nOut_simd;
  const std::unique_ptr<Function> func;
  std::vector<nnReal> initVals;

 public:
  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nWeight.push_back(nOut_simd * (bRecurrent? nInputs + nNeurons : nInputs));
    nBiases.push_back(nNeurons);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(nNeurons);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals.resize(size, 0);
    std::copy(init.begin(), init.end(), initVals.begin());
  }

  BaseLayer(Uint _ID, Uint _nInputs, Uint _nNeurons, std::string funcType,
            bool bRnn, bool bOut, Uint iLink) :
            Layer(_ID, _nNeurons, bOut, false, iLink),
            nInputs(_nInputs), nNeurons(_nNeurons), bRecurrent(bRnn),
            nOut_simd(Utilities::roundUpSimd(_nNeurons)),
            func(makeFunction(funcType))
  {
      spanCompInpGrads = _nInputs;
  }

  std::string printSpecs() const override
  {
    std::ostringstream o;
    o<<"("<<ID<<") "<<func->name()
     <<std::string(bOutput? " output ":" ")
     <<std::string(bRecurrent? "Recurrent-":"")
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
      for (Uint i = 0; i < nInputs; ++i)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; ++o)
          suminp[o] += inputs[i] * W[o];
      }
    }
    if(bRecurrent && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
      for (Uint i = 0; i < nNeurons; ++i)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; ++o)
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
      for(Uint o=0; o<nNeurons; ++o)
        deltas[o] *= func->evalDiff(suminp[o], outval[o]);
    }

    Layer::backward(nInputs, nNeurons, nOut_simd, bRecurrent? nNeurons : 0,
            prev, curr, next, grad, para);
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * func->initFactor(nInputs, nNeurons);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    {
      nnReal* const biases = W->B(ID);
      for(Uint o=0; o<nNeurons; ++o)
        if(initVals.size() != nNeurons) biases[o] = 0;
        else biases[o] = func->inverse(initVals[o]);
    }
    {
      nnReal* const weight = W->W(ID);
      for(Uint i=0; i<nInputs;  ++i)
        for(Uint o=0; o<nNeurons; ++o)
          weight[o +nOut_simd*i] = dis(G);
      //if(std::fabs(fac-1)<nnEPS) orthogonalize(para, gen, init);
    }
    if(bRecurrent)
    {
      nnReal* const weight = W->W(ID) +nOut_simd*nInputs;
      for(Uint i=0; i<nNeurons;  ++i)
        for(Uint o=0; o<nNeurons; ++o)
          weight[o +nOut_simd*i] = dis(G);
    }
  }

  size_t  save(const Parameters * const para,
                          float * tmp) const override
  {
    const nnReal* const bias = para->B(ID);
    const nnReal* const weight = para->W(ID);
    for(Uint i=0; i<nInputs + bRecurrent*nNeurons; ++i)
      for(Uint o=0; o<nNeurons; ++o)
        *(tmp++) = (float) weight[o + nOut_simd * i];
    for (Uint n=0; n<nNeurons; ++n) *(tmp++) = (float) bias[n];
    return nNeurons * (nInputs + bRecurrent*nNeurons + 1);
  }
  size_t restart(const Parameters * const para,
                      const float * tmp) const override
  {
    nnReal* const bias = para->B(ID);
    nnReal* const weight = para->W(ID);
    // restart weights and recurrent weights if any
    for(Uint i=0; i<nInputs + bRecurrent*nNeurons; ++i)
      for(Uint o=0; o<nNeurons; ++o)
        weight[o + nOut_simd * i] = (nnReal) *(tmp++);
    // restart bias
    for (Uint n=0; n<nNeurons; ++n) bias[n] = (nnReal) *(tmp++);
    return nNeurons * (nInputs + bRecurrent*nNeurons + 1);
  }
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h
