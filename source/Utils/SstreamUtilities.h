//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_SstreamUtilities_h
#define smarties_SstreamUtilities_h

#include "Definitions.h"
#include <vector>
#include <sstream>
#include <iomanip> // setprecision

namespace smarties
{

namespace Utilities
{

inline std::string vec2string(const std::vector<std::string> vals)
{
  std::ostringstream o;
  if(!vals.size()) return o.str();
  for (Uint i=0; i<vals.size()-1; ++i) o << vals[i] << " ";
  o << vals[vals.size()-1];
  return o.str();
}

template <typename T>
inline std::string vec2string(const std::vector<T> vals, const int width = -1)
{
  std::ostringstream o;
  if(!vals.size()) return o.str();
  if(width>0) o << std::setprecision(3) << std::fixed;
  for (Uint i=0; i<vals.size()-1; ++i) o << vals[i] << " ";
  o << vals[vals.size()-1];
  return o.str();
}

inline std::string num2str(const int i, const int width)
{
  std::stringstream o;
  o << std::setfill('0') << std::setw(width) << i;
  return o.str();
}

inline void real2SS(std::ostringstream&B,const Real V,const int W, const bool bPos)
{
  B<<" "<<std::setw(W);
  if(std::fabs(V)>= 1e4) B << std::setprecision(std::max(W-7+bPos,0));
  else
  if(std::fabs(V)>= 1e3) B << std::setprecision(std::max(W-6+bPos,0));
  else
  if(std::fabs(V)>= 1e2) B << std::setprecision(std::max(W-5+bPos,0));
  else
  if(std::fabs(V)>= 1e1) B << std::setprecision(std::max(W-4+bPos,0));
  else
                         B << std::setprecision(std::max(W-3+bPos,0));
  B<<std::fixed<<V;
}

} // end namespace smarties
} // end namespace Utilities

#endif // smarties_SstreamUtilities_h
