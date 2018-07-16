//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once


struct MemoryManager
{
  const int nThreads;
  vector<Parameters*> gradients;
  vector<Parameters*> gradients_cuda;
  vector<vector<Activation*>> activations;
  vector<vector<Activation*>> activations_cuda;
  vector<pair<nnReal*, Uint>> workspaces_cudnn;
  vector<cublasHandle_t> handles_cublas;
  vector<cudnnHandle_t> handles_cudnn;
  vector<cudaStream_t> streams_cuda;
};
