//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Builder.h"
#include "Network.h"

namespace smarties
{

Network::Network(const Uint _nInp, const Uint _nOut,
                 std::vector<std::unique_ptr<Layer>>& L,
                 const std::shared_ptr<Parameters>& W) :
  layers(std::move(L)), nInputs(_nInp), nOutputs(_nOut), weights(W) {}

void Network::checkGrads()
{
  const Uint seq_len = 5;
  const nnReal incr = std::sqrt(std::numeric_limits<nnReal>::epsilon());
  const nnReal tol  = std::sqrt(std::numeric_limits<nnReal>::epsilon());;
  printf("Checking grads with increment %e and tolerance %e\n", incr,tol);

  std::vector<std::unique_ptr<Activation>> timeSeries;
  std::shared_ptr<Parameters>     grad = allocParameters();
  std::shared_ptr<Parameters> backGrad = allocParameters(); backGrad->clear();
  std::shared_ptr<Parameters> diffGrad = allocParameters(); diffGrad->clear();
  std::shared_ptr<Parameters>  errGrad = allocParameters();  errGrad->clear();

  std::random_device rd;
  std::mt19937 gen(rd());
  for(Uint t=0; t<seq_len; ++t)
  for(Uint o=0; o<nOutputs; ++o)
  {
    std::vector<std::vector<Real>> inputs(seq_len,std::vector<Real>(nInputs,0));
    std::normal_distribution<nnReal> dis_inp(0, 1);
    for(Uint i=0; i<seq_len; ++i)
      for(Uint j=0; j<nInputs; ++j) inputs[i][j] = dis_inp(gen);

    allocTimeSeries(timeSeries, seq_len);
    for (Uint k=0; k<seq_len; ++k) {
      forward(inputs[k], timeSeries, k);
      std::vector<nnReal> errs(nOutputs, 0);
      if(k==t) {
        errs[o] = -1;
        timeSeries[k]->addOutputDelta(errs);
      }
    }

    grad->clear();
    backProp(timeSeries, t+1, grad.get());

    for (Uint w=0; w<weights->nParams; ++w) {
      nnReal diff = 0;
      const auto copy = weights->params[w];
      //1
      weights->params[w] = copy + incr;
      for (Uint k=0; k<seq_len; ++k) {
        const std::vector<Real> ret = forward(inputs[k], timeSeries, k);
        if(k==t) diff = -ret[o]/(2*incr);
      }
      //2
      weights->params[w] = copy - incr;
      for (Uint k=0; k<seq_len; ++k) {
        const std::vector<Real> ret = forward(inputs[k], timeSeries, k);
        if(k==t) diff += ret[o]/(2*incr);
      }
      //0
      weights->params[w] = copy;

      //const nnReal scale = std::max( fabs(Vgrad[0]->params[w]), fabs(diff) );
      //if (scale < nnEPS) continue;
      const nnReal err = std::fabs(grad->params[w] - diff); //relerr=err/scale;
      if ( err > errGrad->params[w] ) {
        backGrad->params[w] = grad->params[w];
        diffGrad->params[w] = diff;
        errGrad->params[w] = err;
      }
    }
  }

  long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0, sum3 = 0, sumsq3 = 0;
  for (Uint w=0; w<weights->nParams; ++w) {
    if(errGrad->params[w]>tol)
      printf("%lu err:%f, grad:%f, diff:%f, param:%f\n", w, errGrad->params[w],
        backGrad->params[w], diffGrad->params[w], weights->params[w]);

    sum1 += std::fabs(backGrad->params[w]);
    sum2 += std::fabs(diffGrad->params[w]);
    sum3 += std::fabs(errGrad->params[w]);
    sumsq1 += backGrad->params[w] * backGrad->params[w];
    sumsq2 += diffGrad->params[w] * diffGrad->params[w];
    sumsq3 +=  errGrad->params[w] *  errGrad->params[w];
  }

  const long double NW = weights->nParams;
  const auto avg1 = sum1/NW, avg2 = sum2/NW, avg3 = sum3/NW;
  const auto std1 = std::sqrt((sumsq1-sum1*avg1)/NW);
  const auto std2 = std::sqrt((sumsq2-sum2*avg2)/NW);
  const auto std3 = std::sqrt((sumsq3-sum3*avg3)/NW);
  printf("<|grad|>:%Le (std:%Le) <|diff|>:%Le (std:%Le) <|err|>::%Le (std:%Le)\n",
    avg1, std1, avg2, std2, avg3, std3);
  //die("done");
}


void Network::save(const Parameters * const W,
                   const std::string fname,
                   const bool isBackup) const
{
  // if not backup, first write to a temporary file for safety
  const std::string name = fname + ".raw", backname = fname + "_backup.raw";
  FILE * wFile = fopen((isBackup? name : backname).c_str(), "wb");
  float * const buf = Utilities::allocate_ptr<float>(W->nParams);
  size_t totWritten = 0;
  for(const auto & l : layers) {
    const auto written = l->save(W, buf + totWritten);
    totWritten += written;
  }
  fwrite(buf, sizeof(float), totWritten, wFile);
  fflush(wFile); fclose(wFile); free(buf);

  if(not isBackup) Utilities::copyFile(backname, name);
}

int Network::restart(const Parameters * const W,
                     const std::string fname) const
{
  FILE * const wFile = fopen((fname+".raw").c_str(), "rb");
  if(wFile == NULL) {
    printf("Parameters restart file %s not found.\n", (fname+".raw").c_str());
    return 1;
  } else {
    printf("Restarting from file %s.\n", (fname+".raw").c_str());
    fflush(0);
  }
  float* const buf = Utilities::allocate_ptr<float>(W->nParams);
  const size_t wsize = fread(buf, sizeof(float), W->nParams, wFile);

  size_t totWritten = 0;
  for(const auto & l : layers) {
    const auto written = l->restart(W, buf + totWritten);
    totWritten += written; // safety: allocated nParams, not wsize
    assert(totWritten <= W->nParams);
  }
  if(wsize not_eq totWritten)
    _die("Mismatch in restarted file %s; contains:%lu read:%lu.",
      fname.c_str(), wsize, totWritten);
  fclose(wFile); free(buf);

  return 0;
}
#if 0
void Network::dump(const int agentID)
{
  if (not bDump) return;
  char buf[512];
  snprintf(buf, 512, "%07u", (Uint)dump_ID[agentID]);
  string nameNeurons  = "neuronOuts_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameMemories = "cellStates_"+to_string(agentID)+"_"+string(buf)+".dat";
  string nameOut_Mems = "out_states_"+to_string(agentID)+"_"+string(buf)+".dat";
  {
    ofstream out(nameOut_Mems.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameOut_Mems.c_str());
    for (Uint j=0; j<nNeurons; ++j) out << *(mem[agentID]->outvals +j) << " ";
    for (Uint j=0; j<nStates;  ++j) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameNeurons.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameNeurons.c_str());
    for (Uint j=0; j<nNeurons; ++j) out << *(mem[agentID]->outvals +j) << " ";
    out << "\n";
    out.close();
  }
  {
    ofstream out(nameMemories.c_str());
    if(!out.good()) _die("Unable to save into file %s\n", nameMemories.c_str());
    for (Uint j=0; j<nStates;  ++j) out << *(mem[agentID]->ostates +j) << " ";
    out << "\n";
    out.close();
  }
  dump_ID[agentID]++;
}
#endif

} // end namespace smarties
