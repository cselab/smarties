//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Bund.h"
#include <mutex>
#include <sstream>
#include <stdarg.h>

namespace ErrorHandling
{
  static std::mutex warn_mutex;
  enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

  static constexpr Debug_level level = WARNINGS;
  //static constexpr Debug_level level = LEARNERS;
  //static constexpr Debug_level level = SCHEDULER;

//#define PR_ERR(fd,a,...) fprintf(fd,a,__func__,__FILE__,__LINE__,##__VA_ARGS__);
#define FLUSHALL fflush(stdout); fflush(stderr); fflush(0);
#define KILLALL MPI_Abort(MPI_COMM_WORLD, 1);
#define LOCKCOMM std::lock_guard<std::mutex> wlock(ErrorHandling::warn_mutex);

inline static void printfmt(char*const p, const int N, const char*const a, ... )
{
  va_list args;
  va_start (args, a);
  vsnprintf (p, N, a, args);
  va_end (args);
}


// THESE ARE ALL DEFINES ALLOWING PRINTING FILE, FUNC, LINE

#define    die(format)      do { LOCKCOMM  \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL",format); FLUSHALL KILLALL } while(0)

#define   _die(format, ...) do { LOCKCOMM  \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL", BUF); FLUSHALL KILLALL } while(0)

#define  error(format, ...) do { LOCKCOMM \
  int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
  char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " ERROR", BUF); FLUSHALL} while(0)

#define   warn(format)  do { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING",format); FLUSHALL} } while(0)

#define  _warn(format, ...)  do { \
  if(ErrorHandling::level >= ErrorHandling::WARNINGS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING", BUF); FLUSHALL} } while(0)

#define debugS(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::SCHEDULER) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

#define debugE(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::ENVIRONMENT) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

#define debugN(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::NETWORK) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

#define debugC(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::COMMUNICATOR) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

#define _debugL(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::LEARNERS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

#define debugL(format)  do { \
  if(ErrorHandling::level == ErrorHandling::LEARNERS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " LEARNER",format); FLUSHALL} } while(0)

#define debugT(format, ...)  do { \
  if(ErrorHandling::level == ErrorHandling::TRANSITIONS) { LOCKCOMM \
    int wrnk; MPI_Comm_rank(MPI_COMM_WORLD, &wrnk); \
    char BUF[512]; ErrorHandling::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); FLUSHALL} } while(0)

}

template <typename T>
inline std::string print(const std::vector<T> vals, const int width=-1)
{
  std::ostringstream o;
  if(!vals.size()) return o.str();
  if(width>0) o << std::setprecision(3) << std::fixed;
  for (Uint i=0; i<vals.size()-1; i++) o << vals[i] << " ";
  o << vals[vals.size()-1];
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

