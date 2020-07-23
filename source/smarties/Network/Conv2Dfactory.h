//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Builder.h"
#include "Layers/Layer_Conv2D.h"

namespace smarties
{

template
< int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, int Px, int Py, // stride and padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
void makeConv2D(Builder & build,
                const bool bOutput = false,
                const Uint iLink = 1)
{
  assert(KnC*OpX*OpY > 0);
  #ifndef USE_OMPSIMD_BLAS
  build.layers.emplace_back(std::make_unique<
    Mat2ImLayer<InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY> >
      (build.layers.size(), false, iLink) );
  #endif
  build.layers.emplace_back(std::make_unique<
    Conv2DLayer<SoftSign,
                InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY> >
      (build.layers.size(), bOutput, 1) );
}

template
< int InX, int InY, int InC, //input image: x:width, y:height, c:color channels
  int KnX, int KnY, int KnC, //filter:      x:width, y:height, c:color channels
  int Sx, int Sy, int Px, int Py, // stride and padding x/y
  int OpX, int OpY //output img: x:width, y:height, same color channels as KnC
>
bool ifMatchAddConv2D(const Conv2D_Descriptor & DESCR,
                      std::vector<std::unique_ptr<Layer>> & layers,
                      const bool bOutput = false, const Uint iLink = 1)
{
  bool sameInp = DESCR.inpFeatures==InC && DESCR.inpY==InX && DESCR.inpX==InY;
  bool sameOut = DESCR.outFeatures==KnC && DESCR.outY==OpY && DESCR.outX==OpX;
  bool sameFilter  = DESCR.filterx==KnX && DESCR.filtery==KnY;
  bool sameStride  = DESCR.stridex== Sx && DESCR.stridey== Sy;
  bool samePadding = DESCR.paddinx== Px && DESCR.paddiny== Py;
  if( KnC*OpX*OpY == 0 ) die("Requested empty layer.");
  if(sameInp && sameOut && sameFilter && sameStride && samePadding)
  {
    #ifndef USE_OMPSIMD_BLAS
    layers.emplace_back(std::make_unique<
      Mat2ImLayer<InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY> >
        (layers.size(), false, iLink) );
    #endif
    layers.emplace_back(std::make_unique<
      Conv2DLayer<SoftSign,
                  InX,InY,InC, KnX,KnY,KnC, Sx,Sy, Px,Py, OpX,OpY> >
        (layers.size(), bOutput, 1) );
    return true;
  }
  return false;
}
} // end namespace smarties
