//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Layers.h"

class MGULayer: public Layer
{
  const Uint nInputs, nCells;
  const Function* const cell;

 public:
  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override
  {
    //cell, input, forget, output gates all linked to inp and prev LSTM output
    nWeight.push_back(2*nCells * (nInputs + nCells) );
    nBiases.push_back(2*nCells);
  }

  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(2*nCells);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  virtual void biasInitialValues(const vector<nnReal> init) {}

  ~MGULayer() { _dispose_object(cell); }

  MGULayer(Uint _ID, Uint _nInputs, Uint _nCells, string funcType,
    bool bOut, Uint iLink) :  Layer(_ID, _nCells, bOut, false, iLink),
    nInputs(_nInputs), nCells(_nCells), cell(makeFunction(funcType)) {
    spanCompInpGrads = _nInputs;
    if(_nCells % ARY_WIDTH)
      die("hardcoded simd: pick size multiple of 8 for float and 4 for double");
  }

  string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") "<<cell->name()
     <<string(bOutput? " output ":" ")
     <<"MGU Layer of size:"<<nCells
     <<" linked to Layer:"<<ID-link
     <<" of size:"<<nInputs<<"\n";
    return o.str();
  }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    // suminp contains input to all cell inputs and gates
    // only one matrix-vector multiplication
    nnReal* const forget = curr->X(ID);
    nnReal* const cellst = curr->X(ID) + nCells;
    nnReal* const output = curr->Y(ID);
    {
      nnReal* const allinp = curr->X(ID);
      memcpy(allinp, para->B(ID), 2*nCells*sizeof(nnReal));
      const nnReal* const inputs = curr->Y(ID-link);
      const nnReal* const weight = para->W(ID);
      for (Uint i = 0; i < nInputs; i++) {
        const nnReal* const W = weight + (2*nCells)*i;
        #pragma omp simd aligned(allinp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < 2*nCells; o++) allinp[o] += inputs[i] * W[o];
      }
    }

    if(prev not_eq nullptr) {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +(2*nCells)*nInputs;
      for (Uint i=0; i<nCells; i++) {
        const nnReal* const W = weight + (2*nCells)*i;
        #pragma omp simd aligned(forget, inputs, W : VEC_WIDTH)
        for(Uint o=0; o<nCells; o++) forget[o] += W[o] * inputs[i];
      }
      Sigm::_eval(forget, forget, nCells);

      for (Uint i=0; i<nCells; i++) {
        const nnReal* const W = weight +(2*nCells)*i +nCells;
        #pragma omp simd aligned(cellst, forget, inputs, W : VEC_WIDTH)
        for(Uint o=0; o<nCells; o++) cellst[o] += W[o] * inputs[i] * forget[i];
      }
      Tanh::_eval(cellst, cellst, nCells);

      #pragma omp simd aligned(output, forget, inputs, cellst : VEC_WIDTH)
      for (Uint o=0; o<nCells; o++)
        output[o] = forget[o]*inputs[o] + (1-forget[o])*cellst[o];
    }
    else {
      Sigm::_eval(forget, forget, nCells);
      Tanh::_eval(cellst, cellst, nCells);
      #pragma omp simd aligned(output, forget, cellst : VEC_WIDTH)
      for (Uint o=0; o<nCells; o++) output[o] = (1-forget[o])*cellst[o];
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {

    const nnReal* const forget = curr->X(ID);
    const nnReal* const cellst = curr->X(ID) + nCells;
      //const nnReal* const output = curr->Y(ID);
    nnReal* const deltas = curr->E(ID);
    nnReal* const deltaF = curr->E(ID) + nCells;
    nnReal* const deltaC = curr->Y(ID) + nCells;
    nnReal* const prvOut = prev==nullptr? allocate_ptr(nCells) : prev->Y(ID);
    nnReal* const prvErr = prev==nullptr? nullptr : prev->E(ID);

    #pragma omp simd aligned(deltaC, deltas, forget, cellst : VEC_WIDTH)
    for (Uint o=0; o<nCells; o++)
      deltaC[o] = deltas[o] * (1-forget[o]) * (1-cellst[o]*cellst[o]);

    #if 1
    if(prev not_eq nullptr) {
      cblas_dgemv(CblasRowMajor, CblasNoTrans, nCells, nCells, 1,
      para->W(ID) +(2*nCells)*nInputs +nCells, 2*nCells,
      deltaC, 1, 0, deltaF, 1);
    } else memset( deltaF, 0, nCells*sizeof(nnReal) );

    #pragma omp simd aligned(deltaF, prvOut, cellst, deltas, forget : VEC_WIDTH)
    for (Uint o=0; o<nCells; o++)
      deltaF[o] = ((prvOut[o]-cellst[o])*deltas[o] + deltaF[o]*prvOut[o]) *forget[o]*(1-forget[o]);
    #else // more compact and readable:
      for (Uint o=0; o<nCells; o++) {
        nnReal dF = (prvOut[o] - cellst[o]) * deltas[o];
        const nnReal*const weight = para->W(ID) +(2*nCells)*(nInputs+o) +nCells;
        for (Uint k = 0; k < nCells && prev not_eq nullptr; k++)
          dF += deltaC[k] * prvOut[o] * weight[k];
        deltaF[o] = dF * forget[o] * (1-forget[o]);
      }
    #endif

    #if 1
    if(prev not_eq nullptr) {
      nnReal* const tmp = allocate_dirty(nCells);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, nCells, nCells, 1,
      para->W(ID) +(2*nCells)*nInputs +nCells, 2*nCells, deltaC, 1, 0, tmp, 1);

      #pragma omp simd aligned(prvErr, forget, deltas, tmp : VEC_WIDTH)
      for(Uint o=0; o<nCells; o++) prvErr[o] += forget[o]*(deltas[o] + tmp[o]);
      free(tmp);

      cblas_dgemv(CblasRowMajor, CblasNoTrans, nCells, nCells, 1,
      para->W(ID) +(2*nCells)*nInputs, 2*nCells, deltaF, 1, 1, prvErr, 1);
    }
    #else // more compact and readable
    for (Uint o=0; o<nCells && prev not_eq nullptr; o++) {
      prvErr[o] += forget[o] * deltas[o];
      const nnReal* const weight = para->W(ID) +(2*nCells)*(nInputs+o);
      for (Uint k = 0; k < nCells; k++)
        prvErr[o] += deltaF[k]*weight[k] + deltaC[k]*forget[o]*weight[k+nCells];
    }
    #endif

    if( spanCompInpGrads )
    {
            nnReal* const errors = curr->E(ID-link);
      const nnReal* const weight = para->W(ID);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, spanCompInpGrads, nCells, 1,
        weight + startCompInpGrads*2*nCells,          2*nCells,
        deltaF, 1, 1, errors + startCompInpGrads, 1);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, spanCompInpGrads, nCells, 1,
        weight + startCompInpGrads*2*nCells + nCells, 2*nCells,
        deltaC, 1, 1, errors + startCompInpGrads, 1);
    }

    if(prev==nullptr) { free(prvOut); }

    if(grad == nullptr) return;

    {
      nnReal* const grad_b = grad->B(ID);
      #pragma omp simd aligned(grad_b, deltaF, deltaC : VEC_WIDTH)
      for(Uint o=0; o<nCells; o++) {
        grad_b[o]        += deltaF[o];
        grad_b[o+nCells] += deltaC[o];
      }
    }

    {
      const nnReal* const inputs = curr->Y(ID-link);
      for(Uint i=0; i<nInputs;  i++) {
        nnReal* const G = grad->W(ID) + (2*nCells)*i;
        #pragma omp simd aligned(G, inputs, deltaF, deltaC : VEC_WIDTH)
        for(Uint o=0; o<nCells; o++) {
          G[o]        += inputs[i] * deltaF[o];
          G[o+nCells] += inputs[i] * deltaC[o];
        }
      }
    }

    if(prev not_eq nullptr)
    {
      for(Uint i=0; i<nCells; i++) {
        nnReal* const G = grad->W(ID) + 2*nCells * (nInputs + i);
        #pragma omp simd aligned(G, prvOut, deltaF, deltaC, forget : VEC_WIDTH)
        for(Uint o=0; o<nCells; o++) {
          G[o]        += prvOut[i] * deltaF[o];
          G[o+nCells] += prvOut[i] * deltaC[o] * forget[i];
        }
      }
    }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * cell->initFactor(nInputs, nCells);
    uniform_real_distribution<nnReal> dis(-init, init);
    { // forget gate starts open, inp/out gates are closed
     nnReal* const BB = para->B(ID);
     for(Uint o=0*nCells; o<1*nCells; o++) BB[o]=dis(*gen)+LSTM_PRIME_FAC;
     for(Uint o=1*nCells; o<2*nCells; o++) BB[o]=dis(*gen);
    }
    {
     nnReal* const weight = para->W(ID);
     for(Uint w=0; w<2*nCells*(nInputs+nCells); w++) weight[w] = dis(*gen);
    }
  }
};

    // ft = sf (wf xt + uf ho)
    // ct = sr (wh xt + uh ft ho)
    // ht = ft*ho + (1-ft)*ct
    // dc = e*(1-ft)*ct'
    // df = ((ho-rt)*e + dc * uh^T *ho )*ft'
    // dh = e*ft + uf * df + ft * uh * dc
