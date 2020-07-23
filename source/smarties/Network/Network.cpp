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
  layers(std::move(L)), nInputs(_nInp), nOutputs(_nOut), weights(W)
{}


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
