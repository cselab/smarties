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
  /*
  const Uint seq_len = 5;
  const nnReal incr = std::cbrt(std::numeric_limits<nnReal>::epsilon())
  const nnReal tol  = incr;
  printf("Checking grads with increment %e and tolerance %e\n", incr,tol);

  std::vector<Activation*> timeSeries;
  if(Vgrad.size() < 4) die("I'm the worst, just use 4 threads and forgive me");
  Vgrad[1]->clear(); Vgrad[2]->clear(); Vgrad[3]->clear();
  std::random_device rd;
  std::mt19937 gen(rd());
  for(Uint t=0; t<seq_len; ++t)
  for(Uint o=0; o<nOutputs; ++o)
  {
    vector<vector<Real>> inputs(seq_len, vector<Real>(nInputs,0));
    prepForBackProp(timeSeries, seq_len);
    Vgrad[0]->clear();
    normal_distribution<nnReal> dis_inp(0, 1);
    for(Uint i=0; i<seq_len; ++i)
      for(Uint j=0; j<nInputs; ++j) inputs[i][j] = dis_inp(gen);

    for (Uint k=0; k<seq_len; ++k) {
      predict(inputs[k], timeSeries, k);
      vector<nnReal> errs(nOutputs, 0);
      if(k==t) {
        errs[o] = -1;
        timeSeries[k]->addOutputDelta(errs);
      }
    }
    backProp(timeSeries, t+1, Vgrad[0]);

    for (Uint w=0; w<weights->nParams; ++w) {
      nnReal diff = 0;
      const auto copy = weights->params[w];
      //1
      weights->params[w] += incr;
      for (Uint k=0; k<seq_len; ++k) {
        const vector<Real> ret = predict(inputs[k], timeSeries, k);
        if(k==t) diff = -ret[o]/(2*incr);
      }
      //2
      weights->params[w] = copy - incr;
      for (Uint k=0; k<seq_len; ++k) {
        const vector<Real> ret = predict(inputs[k], timeSeries, k);
        if(k==t) diff += ret[o]/(2*incr);
      }
      //0
      weights->params[w] = copy;

      //const nnReal scale = std::max( fabs(Vgrad[0]->params[w]), fabs(diff) );
      //if (scale < nnEPS) continue;
      const nnReal err = fabs(Vgrad[0]->params[w]-diff);//relerr=err/scale;
      // if error now is bigger or if equal but grad magnitude is greater
      if( err>Vgrad[2]->params[w] || ( err>=Vgrad[2]->params[w] &&
         std::fabs(Vgrad[1]->params[w]) < std::fabs(Vgrad[0]->params[w]) ) ) {
        Vgrad[1]->params[w] = Vgrad[0]->params[w];
        Vgrad[2]->params[w] = err;
        Vgrad[3]->params[w] = diff;
      }
    }
  }

  long double sum1 = 0, sumsq1 = 0, sum2 = 0, sumsq2 = 0;
  for (Uint w=0; w<weights->nParams; ++w) {
    if(Vgrad[2]->params[w]>tol)
    cout<<w<<" err:"<<Vgrad[2]->params[w]<<", grad:"<<Vgrad[1]->params[w]
        <<" diff:"<<Vgrad[3]->params[w]<<" param:"<<weights->params[w]<<endl;

    sum1+=std::fabs(Vgrad[1]->params[w]); sum2+=std::fabs(Vgrad[2]->params[w]);
    sumsq1 += Vgrad[1]->params[w]*Vgrad[1]->params[w];
    sumsq2 += Vgrad[2]->params[w]*Vgrad[2]->params[w];
  }

  long double NW = weights->nParams, avg1 = sum1/NW, avg2 = sum2/NW;
  auto std1=sqrt((sumsq1-sum1*avg1)/NW), std2=sqrt((sumsq2-sum2*avg2)/NW);
  cout<< "Abs gradient avg:" <<avg1<<" std:"<<std1
      <<" Abs error avg:"<<avg2<<" std:"<<std2<<endl;
  deallocateUnrolledActivations(&timeSeries);
  Vgrad[0]->clear(); Vgrad[1]->clear(); Vgrad[2]->clear(); Vgrad[3]->clear();
  die("done");
  */
}

#if 0
void Network::dump(const int agentID)
{
  if (not bDump) return;
  char buf[500];
  sprintf(buf, "%07u", (Uint)dump_ID[agentID]);
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
