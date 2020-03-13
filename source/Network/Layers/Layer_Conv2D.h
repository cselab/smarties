//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Conv2DLayer_h
#define smarties_Conv2DLayer_h

#include "Layers.h"

namespace smarties
{

#define ALIGNSPEC __attribute__(( aligned(VEC_WIDTH) ))

// Im2MatLayer gets as input an image of sizes InX * InY * InC
// and prepares the output for convolution with a filter of size KnY * KnX * KnC
// and output an image of size OpY * OpX * KnC
template
< typename func,
  int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, int Px, int Py, // stride and padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Conv2DLayer: public Layer
{

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nBiases.push_back(out_size);
    nWeight.push_back(KnC * InC * KnY * KnX);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(out_size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }


  Conv2DLayer(int _ID, bool bOut, Uint iLink) :
    Layer(_ID, out_size, bOut, false, iLink) {
    spanCompInpGrads = inp_size;
    static_assert(InC>0 && InY>0 && InX>0, "Invalid input image size");
    static_assert(KnC>0 && KnY>0 && KnX>0, "Invalid kernel size");
    static_assert(OpY>0 && OpX>0, "Invalid output image size");
    static_assert(Px>=0 && Py>=0, "Invalid padding");
    static_assert(Sx>0 && Sy>0, "Invalid stride");
  }

  std::string printSpecs() const override {
    std::ostringstream o;
    auto nonlin = std::make_unique<func>();
    o<<"("<<ID<<") "<<nonlin->name()
     <<"Conv Layer with Input:["<<InC<<"x"<<InY<<"x"<<InX
     <<"] Filter:["<<InC<<"x"<<KnC<<"x"<<KnY<<"x"<<KnX
     <<"] Output:["<<KnC<<"x"<<OpY<<"x"<<OpX
     <<"] Stride:["<<Sx<<"x"<<Sy <<"] Padding:["<<Px<<"x"<<Py
     <<"] linked to Layer:"<<ID-link<<"\n";
    return o.str();
  }

  void backward_bias(
          nnReal* const dLdOut,          nnReal* const dLdBias,
    const nnReal* const LinearOut, const nnReal* const NonLinOut) const
  { // premultiply with derivative of non-linearity & grad of bias
    #pragma omp simd aligned(dLdOut, LinearOut, NonLinOut, dLdBias : VEC_WIDTH)
    for(int o = 0; o < out_size; ++o) {
      dLdOut [o] *= func::_evalDiff(LinearOut[o], NonLinOut[o]);
      dLdBias[o] += dLdOut[o];
    }
  }

  #ifdef USE_OMPSIMD_BLAS

  using Input  = ALIGNSPEC nnReal[InC][InY][InX];
  using Kernel = ALIGNSPEC nnReal[KnC][InC][KnY][KnX];
  using Output = ALIGNSPEC nnReal[KnC][OpY][OpX];
  static constexpr int inp_size = InC * InX * InY;
  static constexpr int out_size = KnC * OpY * OpX;

  void forward(const Activation*const prev,
               const Activation*const curr,
               const Parameters*const para) const override
  {
    // Convert pointers to a reference to multi dim arrays for easy access:
    const Input  & __restrict__ INP = * (Input *) curr->Y(ID-link);
          Output & __restrict__ OUT = * (Output*) curr->X(ID);
    const Kernel & __restrict__ K   = * (Kernel*) para->W(ID);

    memcpy(curr->X(ID), para->B(ID), out_size * sizeof(nnReal));

    for (int fc = 0; fc < KnC; ++fc) for (int ic = 0; ic < InC; ++ic)
    for (int oy = 0; oy < OpY; ++oy) for (int fy = 0; fy < KnY; ++fy)
    for (int ox = 0; ox < OpX; ++ox) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      OUT[fc][oy][ox] += K[fc][ic][fy][fx] * INP[ic][iy][ix];
    }

    func::_eval(curr->X(ID), curr->Y(ID), out_size);
    // memset 0 because padding and forward assumes overwrite
    memset(curr->Y(ID), 0, out_size * sizeof(nnReal) );
  }

  void backward(const Activation*const prev,
                const Activation*const curr,
                const Activation*const next,
                const Parameters*const grad,
                const Parameters*const para) const override
  {
    // premultiply with derivative of non-linearity & grad of bias:
    backward_bias(curr->E(ID), grad->B(ID), curr->X(ID), curr->Y(ID));

          Input  & __restrict__ dLdINP = * (Input *) curr->E(ID-link);
    const Output & __restrict__ dLdOUT = * (Output*) curr->E(ID);
          Kernel & __restrict__ dLdK   = * (Kernel*) grad->W(ID);
    const Input  & __restrict__ INP    = * (Input *) curr->Y(ID-link);
    const Kernel & __restrict__ K      = * (Kernel*) para->W(ID);
    // no memset 0 of grad because backward assumed additive
    for (int fc = 0; fc < KnC; ++fc) for (int ic = 0; ic < InC; ++ic)
    for (int oy = 0; oy < OpY; ++oy) for (int fy = 0; fy < KnY; ++fy)
    for (int ox = 0; ox < OpX; ++ox) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      dLdK[fc][ic][fy][fx] += dLdOUT[fc][oy][ox] * INP[ic][iy][ix];
      dLdINP[ic][iy][ix]   += dLdOUT[fc][oy][ox] * K[fc][ic][fy][fx];
    }
  }

  #else // USE_OMPSIMD_BLAS

  static constexpr int inp_size = InC*KnY*KnX*OpY*OpX;
  static constexpr int out_size = KnC*OpY*OpX;

  void forward(const Activation*const prev,
               const Activation*const curr,
               const Parameters*const para) const override
  {
    memcpy(curr->X(ID), para->B(ID), out_size * sizeof(nnReal));
    // [KnC][OpY*OpX] = [KnC][InC*KnY*KnX] * [InC*KnY*KnX][OpY*OpX]
    static constexpr int outRow = KnC, nInner = InC*KnY*KnX, outCol = OpY*OpX;
    #ifdef USE_OMPSIMD_BLAS
    GEMMomp<outRow, nInner, nInner, outCol, false, false>
      ( para->W(ID), curr->Y(ID-link), curr->X(ID) );
    #else
    SMARTIES_gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, outRow, outCol, nInner,
      1, para->W(ID), nInner, curr->Y(ID-link), outCol, 1, curr->X(ID), outCol);
    #endif
    func::_eval(curr->X(ID), curr->Y(ID), KnC * OpY * OpX);
  }

  void backward(const Activation*const prev,
                const Activation*const curr,
                const Activation*const next,
                const Parameters*const grad,
                const Parameters*const para) const override
  {
    // premultiply with derivative of non-linearity & grad of bias:
    backward_bias(curr->E(ID), grad->B(ID), curr->X(ID), curr->Y(ID));
    {
      static constexpr int outRow = KnC, outCol = InC*KnY*KnX, nInner = OpY*OpX;
      // Compute gradient of error wrt to kernel parameters:
      // [KnC][InC*KnY*KnX] = [KnC][OpY*OpX] * ([InC*KnY*KnX][OpY*OpX])^T
      #ifdef USE_OMPSIMD_BLAS
      GEMMomp<outRow,nInner, outCol, nInner, false, true>
        ( curr->E(ID), curr->Y(ID-link), grad->W(ID) );
      #else
      SMARTIES_gemm(CblasRowMajor, CblasNoTrans, CblasTrans, outRow, outCol, nInner,
      1, curr->E(ID), nInner, curr->Y(ID-link), nInner, 1, grad->W(ID), outCol);
      #endif
    }
    // if this is the second layer then this would compute useless dLossDPixels
    if(ID>2) {
      // Compute gradient of error wrt to output of previous layer:
      //[InC*KnY*KnX][OpY*OpX] = ([KnC][InC*KnY*KnX])^T [KnC][OpY*OpX]
      static constexpr int outRow = InC*KnY*KnX, outCol = OpY*OpX, nInner = KnC;
      #ifdef USE_OMPSIMD_BLAS
      GEMMomp<nInner,outRow, nInner, outCol, true, false>
        ( para->W(ID), curr->E(ID), curr->E(ID-link) );
      #else
      SMARTIES_gemm(CblasRowMajor, CblasTrans, CblasNoTrans, outRow, outCol, nInner,
      1, para->W(ID), outRow, curr->E(ID), outCol, 1, curr->E(ID-link), outCol);
      #endif
    }
  }

  #endif // USE_OMPSIMD_BLAS

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const int nAdded = InC * KnX * KnY;
    const nnReal init = fac * func::_initFactor(nAdded, KnC);
    std::uniform_real_distribution<nnReal> dis(-init, init);
    nnReal* const biases = W->B(ID);
    nnReal* const weight = W->W(ID);
    assert(W->NB(ID) == out_size);
    assert(W->NW(ID) == KnX * KnY * KnC * InC);
    for(Uint o=0; o < W->NB(ID); ++o) biases[o] = 0;
    for(Uint o=0; o < W->NW(ID); ++o) weight[o] = dis(G);
  }

  size_t  save(const Parameters * const para,
                          float * tmp) const override
  {
    const nnReal* const bias = para->B(ID);
    const nnReal* const weight = para->W(ID);
    for (Uint n=0; n<para->NW(ID); ++n) *(tmp++) = (float) weight[n];
    for (Uint n=0; n<para->NB(ID); ++n) *(tmp++) = (float) bias[n];
    return para->NB(ID) + para->NW(ID);
  }
  size_t restart(const Parameters * const para,
                      const float * tmp) const override
  {
    nnReal* const bias = para->B(ID);
    nnReal* const weight = para->W(ID);
    for (Uint n=0; n<para->NW(ID); ++n) weight[n] = (nnReal) *(tmp++);
    for (Uint n=0; n<para->NB(ID); ++n) bias[n] = (nnReal) *(tmp++);
    return para->NB(ID) + para->NW(ID);
  }
};

#ifndef USE_OMPSIMD_BLAS
// Im2MatLayer gets as input an image of sizes InX * InY * InC
// and prepares the output for convolution with a filter of size KnY * KnX * KnC
// and output an image of size OpY * OpX * KnC
template
< int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, int Px, int Py, // stride and padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
struct Mat2ImLayer: public Layer
{
  using Input  = ALIGNSPEC nnReal[InC][InY][InX];
  using Output = ALIGNSPEC nnReal[InC][KnY][KnX][OpY][OpX];
  static constexpr int inp_size = InC*InX*InY;
  static constexpr int out_size = InC*KnY*KnX*OpY*OpX;

  void requiredParameters(std::vector<Uint>& nWeight,
                          std::vector<Uint>& nBiases ) const override {
    nBiases.push_back(0);
    nWeight.push_back(0);
  }
  void requiredActivation(std::vector<Uint>& sizes,
                          std::vector<Uint>& bOutputs,
                          std::vector<Uint>& bInputs) const override {
    sizes.push_back(out_size);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const std::vector<Real> init) override { }

  Mat2ImLayer(int _ID, bool bOut, Uint iLink):
    Layer(_ID, out_size, bOut, false, iLink)
  {
    spanCompInpGrads = inp_size;
    static_assert(InC>0 && InY>0 && InX>0, "Invalid input image size");
    static_assert(KnC>0 && KnY>0 && KnX>0, "Invalid kernel size");
    static_assert(OpY>0 && OpX>0, "Invalid output image size");
    static_assert(Px>=0 && Py>=0, "Invalid padding");
    static_assert(Sx>0 && Sy>0, "Invalid stride");
  }

  std::string printSpecs() const override {
    std::ostringstream o;
    o<<"("<<ID<<") Mat2Im Layer with Input:["<<InC<<"x"<<InY<<"x"<<InX
     <<"] Filter:["<<InC<<"x"<<KnC<<"x"<<KnY<<"x"<<KnX
     <<"] Output:["<<KnC<<"x"<<OpY<<"x"<<OpX
     <<"] Stride:["<<Sx<<"x"<<Sy <<"] Padding:["<<Px<<"x"<<Py
     <<"] linked to Layer:"<<ID-link<<"\n";
    return o.str();
  }

  void forward(const Activation*const prev,
               const Activation*const curr,
               const Parameters*const para) const override
  {
    // Convert pointers to a reference to multi dim arrays for easy access:
    const Input  & __restrict__ INP = * (Input *) curr->Y(ID-link);
          Output & __restrict__ OUT = * (Output*) curr->Y(ID);
    // memset 0 because padding and forward assumes overwrite
    memset(curr->Y(ID), 0, out_size * sizeof(nnReal) );

    for (int ic = 0; ic < InC; ++ic)
    for (int oy = 0; oy < OpY; ++oy) for (int fy = 0; fy < KnY; ++fy)
    for (int ox = 0; ox < OpX; ++ox) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      OUT[ic][fy][fx][oy][ox] = INP[ic][iy][ix];
    }
  }

  void backward(const Activation*const prev,
                const Activation*const curr,
                const Activation*const next,
                const Parameters*const grad,
                const Parameters*const para) const override
  {
    // if this is the first layer then this would compute useless dLossDPixels
    if(ID==1) return;
          Input  & __restrict__ dLdINP = * (Input *) curr->E(ID-link);
    const Output & __restrict__ dLdOUT = * (Output*) curr->E(ID);
    // no memset 0 of grad because backward assumed additive
    for (int ic = 0; ic < InC; ++ic) //loop over inp feature maps
    for (int oy = 0; oy < OpY; ++oy) for (int fy = 0; fy < KnY; ++fy)
    for (int ox = 0; ox < OpX; ++ox) for (int fx = 0; fx < KnX; ++fx) {
      //starting position along input map for convolution with kernel
      const int ix = ox*Sx - Px + fx; //index along input map of
      const int iy = oy*Sy - Py + fy; //the convolution op
      //padding: skip addition if outside input boundaries
      if (ix < 0 || ix >= InX || iy < 0 || iy >= InY) continue;
      dLdINP[ic][iy][ix] += dLdOUT[ic][fy][fx][oy][ox];
    }
  }

  void initialize(std::mt19937& G, const Parameters*const W,
                  Real initializationFac) const override { }
  size_t   save(const Parameters * const para,
                           float * tmp) const override { return 0; }
  size_t restart(const Parameters * const para,
                      const float * tmp) const override { return 0; }
};

#endif // USE_OMPSIMD_BLAS

#undef ALIGNSPEC

} // end namespace smarties
#endif // smarties_Quadratic_term_h
