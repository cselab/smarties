//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Diego Rossinelli.
//

#pragma once

#include <assert.h>
#undef min
#undef max
#include <vector>
#undef min
#undef max
#include <map>
#include <string>
#include <stdio.h>
#include <stack>
#include <sys/time.h>

using namespace std;

#include <sys/time.h>

const bool bVerboseProfiling = false;

class ProfileAgent
{
  //  typedef tbb::tick_count ClockTime;
  typedef timeval ClockTime;

  enum ProfileAgentState{ ProfileAgentState_Created, ProfileAgentState_Started, ProfileAgentState_Stopped};

  ClockTime m_tStart, m_tEnd;
  ProfileAgentState m_state;
  long double m_dAccumulatedTime;
  int m_nMeasurements;
  int m_nMoney;

  static inline void _getTime(ClockTime& time)
  {
    //time = tick_count::now();
    gettimeofday(&time, NULL);
  }

  static inline double _getElapsedTime(const ClockTime&tS, const ClockTime&tE)
  {
    return (tE.tv_sec - tS.tv_sec) + 1e-6 * (tE.tv_usec - tS.tv_usec);
    //return (tE - tS).seconds();
  }

  void _reset()
  {
    m_tStart = ClockTime();
    m_tEnd = ClockTime();
    m_dAccumulatedTime = 0;
    m_nMeasurements = 0;
    m_nMoney = 0;
    m_state = ProfileAgentState_Created;
  }

public:

  ProfileAgent():m_tStart(), m_tEnd(), m_state(ProfileAgentState_Created),
  m_dAccumulatedTime(0), m_nMeasurements(0), m_nMoney(0) {}

  void start()
  {
    assert(m_state == ProfileAgentState_Created || m_state == ProfileAgentState_Stopped);

    if (bVerboseProfiling) {printf("start\n");}

    _getTime(m_tStart);

    m_state = ProfileAgentState_Started;
  }

  void stop(int nMoney=0)
  {
    assert(m_state == ProfileAgentState_Started);

    if (bVerboseProfiling) {printf("stop\n");}

    _getTime(m_tEnd);
    m_dAccumulatedTime += _getElapsedTime(m_tStart, m_tEnd);
    m_nMeasurements++;
    m_nMoney += nMoney;
    m_state = ProfileAgentState_Stopped;
  }

  friend class Profiler;
};

struct ProfileSummaryItem
{
  string sName;
  double dTime;
  double dAverageTime;
  int nMoney;
  int nSamples;

  ProfileSummaryItem(string sName_, double dTime_, int nMoney_, int nSamples_):
    sName(sName_),dTime(dTime_),dAverageTime(dTime_/nSamples_),nMoney(nMoney_),nSamples(nSamples_){}
};


class Profiler
{
protected:

  map<string, ProfileAgent*> m_mapAgents;
  stack<string> m_mapStoppedAgents;

public:
  inline void push_start(string sAgentName)
  {
    //if (m_mapStoppedAgents.size() > 0)
    //  getAgent(m_mapStoppedAgents.top()).stop();

    m_mapStoppedAgents.push(sAgentName);
    getAgent(sAgentName).start();
  }

  inline void stop_start(string sAgentName)
  {
    stop_all();
    push_start(sAgentName);
  }
  inline void check_start(string sAgentName)
  {
    if(m_mapStoppedAgents.size() > 0)
    {
      stop_all();
      push_start(sAgentName);
    }
  }

  inline void pop_stop()
  {
    if(m_mapStoppedAgents.size() > 0)
    {
      string sCurrentAgentName = m_mapStoppedAgents.top();
      getAgent(sCurrentAgentName).stop();
      m_mapStoppedAgents.pop();
    }
    //if (m_mapStoppedAgents.size() == 0) return;
    //getAgent(m_mapStoppedAgents.top()).start();
  }

  inline void stop_all()
  {
    while(m_mapStoppedAgents.size() > 0)
    {
      string sCurrentAgentName = m_mapStoppedAgents.top();
      getAgent(sCurrentAgentName).stop();
      m_mapStoppedAgents.pop();
    }
    //if (m_mapStoppedAgents.size() == 0) return;
    //getAgent(m_mapStoppedAgents.top()).start();
  }

  void clear()
  {
    for(auto &item : m_mapAgents) {
      delete item.second;
      item.second = nullptr;
    }

    m_mapAgents.clear();
  }

  Profiler(): m_mapAgents(){}

  ~Profiler()
  {
    clear();
  }

  void printSummary(FILE *outFile=NULL)
  {
    stop_all();
    vector<ProfileSummaryItem> v = createSummary();

    double dTotalTime = 0;
    double dTotalTime2 = 0;
    for(const auto &item : v) dTotalTime += item.dTime;

    for(const auto &item : v) dTotalTime2 += item.dTime - item.nSamples*1.30e-6;

    for(const auto &item : v) {
      const double avgTime = item.dAverageTime;
      const double frac1 = 100*item.dTime/dTotalTime;
      const double frac2 = 100*(item.dTime- item.nSamples*1.3e-6)/dTotalTime2;
      if(frac1 < 1 && frac2 < 1) continue;
      printf("[%15s]: \t%02.0f-%02.0f%%\t%03.3e (%03.3e) s\t%03.3f (%03.3f) s\t(%d samples)\n",
        item.sName.data(), frac1, frac2, avgTime, avgTime-1.30e-6, item.dTime,
        item.dTime- item.nSamples*1.30e-6, item.nSamples);
      if(outFile)fprintf(outFile,"[%15s]: \t%02.2f%%\t%03.3f s\t(%d samples)\n",
          item.sName.data(), 100*item.dTime/dTotalTime, avgTime, item.nSamples);
    }

    printf("[Total time]: \t%f\n", dTotalTime);
    if (outFile) fprintf(outFile,"[Total time]: \t%f\n", dTotalTime);
    if (outFile) fflush(outFile);
    if (outFile) fclose(outFile);
  }

  vector<ProfileSummaryItem> createSummary(bool bSkipIrrelevantEntries=true) const
  {
    vector<ProfileSummaryItem> result;
    result.reserve(m_mapAgents.size());

    for(const auto &item : m_mapAgents) {
      const ProfileAgent& agent = *item.second;
      if (!bSkipIrrelevantEntries || agent.m_dAccumulatedTime>1e-3)
        result.push_back(ProfileSummaryItem(item.first, agent.m_dAccumulatedTime, agent.m_nMoney, agent.m_nMeasurements));
    }
    return result;
  }

  void reset()
  {
    stop_all();
    for(const auto &item : m_mapAgents) item.second->_reset();
  }

  ProfileAgent& getAgent(string sName)
  {
    if (bVerboseProfiling) {printf("%s ", sName.data());}

    map<string, ProfileAgent*>::const_iterator it = m_mapAgents.find(sName);

    const bool bFound = it != m_mapAgents.end();

    if (bFound) return *it->second;

    ProfileAgent * agent = new ProfileAgent();

    m_mapAgents[sName] = agent;

    return *agent;
  }

  friend class ProfileAgent;
};
