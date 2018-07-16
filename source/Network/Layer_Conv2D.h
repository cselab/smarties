//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Layers.h"

template<typename func, //nonlinearity
int In_X, int In_Y, int In_C, //input image: x:width, y:height, c:color channels
int Kn_X, int Kn_Y, int Kn_C,  //filter: x:width, y:height, c:color channels
int Sx, int Sy, //stride x/y
int OutX, int OutY, int OutC=Kn_C, //output image
int Px=0, int Py=0>
class ConvLayer : public Layer
{
 public:
  const Uint link;
  ConvLayer(int _ID, bool bOut, Uint iLink) :
   Layer(_ID, OutX*OutY*Kn_C, bOut), link(iLink) {
    spanCompInpGrads = In_X * In_Y * In_C;
    assert(In_X>0 && In_Y>0 && In_C>0);
    assert(Kn_X>0 && Kn_Y>0 && Kn_C>0);
    assert(Sx>0 && Sy>0 && OutX>0 && OutY>0);
    assert((In_X-Kn_X) % Sx == 0);
    assert((In_Y-Kn_Y) % Sy == 0);
    assert(OutC == Kn_C || OutC == 2*Kn_C);
  }

  string printSpecs() const override {
    std::ostringstream o;
    func* nonlin = new func();
    o<<"("<<ID<<") "<<nonlin->name()
     <<"Conv Layer with Input:["<<In_X<<" "<<In_Y<<" "<<In_C
     <<"] Filter:["<<Kn_X<<" "<<Kn_Y<<" "<<Kn_C
     <<"] Output:["<<OutX<<" "<<OutY
     <<"] Stride:["<<Sx<<" "<<Sy
     <<"] Padding:["<<Px<<" "<<Py
     <<"] linked to Layer:"<<ID-link<<"\n";
    delete nonlin;
    return o.str();
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nBiases.push_back(OutX * OutY * Kn_C);
    nWeight.push_back(Kn_X * Kn_Y * Kn_C * In_C);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(OutX*OutY*Kn_C);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<nnReal> init) override { }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    __attribute__((aligned(32))) const nnReal (&convInp)[In_Y][In_X][In_C] =
         * reinterpret_cast < nnReal ( * __restrict__ ) [In_Y][In_X][In_C]> (
      curr->Y(ID-link) );

    __attribute__((aligned(32))) nnReal buf[OutY][OutX][Kn_Y][Kn_X][In_C];
    memcpy(curr->X(ID), para->B(ID), OutX*OutY*Kn_C*sizeof(nnReal));

    for(int oy=0, iy0= -Py; oy<OutY; oy++, iy0+=Sy) //2loops over output images
    for(int ox=0, ix0= -Px; ox<OutX; ox++, ix0+=Sx)
     for(int fy=0, iy=iy0; fy<Kn_Y; fy++, iy++) //2loops for filter width/height
     for(int fx=0, ix=ix0; fx<Kn_X; fx++, ix++)
      //padding: skip addition if outside input boundaries
      if ( ix < 0 || ix >= In_X || iy < 0 || iy >= In_Y) continue;
      else
      memcpy(&buf[oy][ox][fy][fx][0],&convInp[iy][ix][0], In_C*sizeof(nnReal));

    static const int mm_outRow = OutY * OutX;
    static const int mm_nInner = Kn_Y * Kn_X * In_C;
    static const int mm_outCol = Kn_C;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      mm_outRow, mm_outCol, mm_nInner, 1, &buf[0][0][0][0][0], mm_nInner,
      para->W(ID), mm_outCol, 1, curr->X(ID), mm_outCol);

    //apply per-pixel non-linearity:
    func::_eval(curr->X(ID), curr->Y(ID), OutX*OutY*Kn_C);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    { // premultiply with derivative of non-linearity
            nnReal* const deltas = curr->E(ID);
            nnReal* const grad_b = grad->B(ID);
      const nnReal* const suminp = curr->X(ID);
      const nnReal* const outval = curr->Y(ID);
      for(Uint o=0; o<OutX*OutY*Kn_C; o++) {
        deltas[o] *= func::_evalDiff(suminp[o], outval[o]);
        grad_b[o] += deltas[o];
      }
    }
    __attribute__((aligned(32))) const nnReal (&convErr)[OutY][OutX][Kn_C] =
         * reinterpret_cast < nnReal ( * __restrict__ ) [OutY][OutX][Kn_C]> (
      curr->E(ID) );

    __attribute__((aligned(32))) const nnReal (&convInp)[In_Y][In_X][In_C] =
         * reinterpret_cast < nnReal ( * __restrict__ ) [In_Y][In_X][In_C]> (
      curr->Y(ID-link) );
    __attribute__((aligned(32)))       nnReal (&inpErr )[In_Y][In_X][In_C] =
         * reinterpret_cast < nnReal ( * __restrict__ ) [In_Y][In_X][In_C]> (
      curr->E(ID-link) );

    __attribute__((aligned(32))) const nnReal (&K)[Kn_Y][Kn_X][In_C][Kn_C] =
     * reinterpret_cast <nnReal( * __restrict__ ) [Kn_Y][Kn_X][In_C][Kn_C]> (
      para->W(ID) );
    __attribute__((aligned(32)))       nnReal (&G)[Kn_Y][Kn_X][In_C][Kn_C] =
     * reinterpret_cast <nnReal( * __restrict__ ) [Kn_Y][Kn_X][In_C][Kn_C]> (
      grad->W(ID) );

    for(int oy=0, iy0= -Py; oy<OutY; oy++, iy0+=Sy) //2loops over output images
    for(int ox=0, ix0= -Px; ox<OutX; ox++, ix0+=Sx)
     for(int fy=0, iy=iy0; fy<Kn_Y; fy++, iy++) //2loops for filter width/height
     for(int fx=0, ix=ix0; fx<Kn_X; fx++, ix++)
       //padding: skip addition if outside input boundaries
       if ( ix < 0 || ix >= In_X || iy < 0 || iy >= In_Y) continue;
       else
        for(int ic=0; ic<In_C; ic++) //loop over inp feature maps
         for(int fc=0; fc<Kn_C; fc++) {
           G[fy][fx][ic][fc] += convErr[oy][ox][fc] * convInp[iy][ix][ic];
           inpErr[iy][ix][ic] += convErr[oy][ox][fc] * K[fy][fx][ic][fc];
         }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const int nAdded = Kn_X*Kn_Y*In_C;
    const nnReal init = fac * func::_initFactor(nAdded, Kn_C);
    uniform_real_distribution<nnReal> dis(-init, init);
    nnReal* const biases = para->B(ID);
    nnReal* const weight = para->W(ID);
    assert(para->NB(ID) == OutX * OutY * Kn_C);
    assert(para->NW(ID) == Kn_X * Kn_Y * Kn_C * In_C);
    for(Uint o=0; o<para->NB(ID); o++) biases[o] = dis(*gen);
    for(Uint o=0; o<para->NW(ID); o++) weight[o] = dis(*gen);
  }

  void orthogonalize(const Parameters*const para) const {}



};
