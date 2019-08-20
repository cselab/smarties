//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Warnings_h
#define smarties_Warnings_h

#include "MPIUtilities.h"
#include <mutex>
//#include <sstream>
#include <stdarg.h>

namespace smarties
{
namespace Warnings
{
static std::mutex warn_mutex;
enum Debug_level { SILENT, WARNINGS, SCHEDULER, ENVIRONMENT, NETWORK, COMMUNICATOR, LEARNERS, TRANSITIONS };

static constexpr Debug_level level = WARNINGS;
//static constexpr Debug_level level = LEARNERS;
//static constexpr Debug_level level = SCHEDULER;

static inline void flushAll() { fflush(stdout); fflush(stderr); fflush(0); }

static inline void abortAll() { MPI_Abort(MPI_COMM_WORLD, 1); }

#define SMARTIES_LOCKCOMM std::lock_guard<std::mutex> wlock(smarties::Warnings::warn_mutex)

inline static void printfmt(char*const p, const int N, const char*const a, ... )
{
  va_list args;
  va_start (args, a);
  vsnprintf (p, N, a, args);
  va_end (args);
}
// THESE ARE ALL DEFINES ALLOWING PRINTING FILE, FUNC, LINE

#define    die(format)      do { \
  SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL",format); smarties::Warnings::flushAll(); smarties::Warnings::abortAll(); } while(0)

#define   _die(format, ...) do { \
  SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
  char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " FATAL", BUF); smarties::Warnings::flushAll(); smarties::Warnings::abortAll(); } while(0)

#define  error(format, ...) do { \
  SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
  char BUF[512]; Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
  " ERROR", BUF); smarties::Warnings::flushAll(); } while(0)

#define   warn(format)  do { \
  if(Warnings::level >= smarties::Warnings::WARNINGS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING",format); smarties::Warnings::flushAll(); } } while(0)

#define  _warn(format, ...)  do { \
  if(Warnings::level >= smarties::Warnings::WARNINGS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " WARNING", BUF); smarties::Warnings::flushAll(); } } while(0)

#define debugS(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::SCHEDULER) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stdout,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

#define debugE(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::ENVIRONMENT) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

#define debugN(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::NETWORK) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

#define debugC(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::COMMUNICATOR) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

#define _debugL(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

#define debugL(format)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " LEARNER",format); smarties::Warnings::flushAll(); } } while(0)

#define debugT(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::TRANSITIONS) { \
    SMARTIES_LOCKCOMM; const auto wrnk = smarties::MPIworldRank(); \
    char BUF[512]; smarties::Warnings::printfmt(BUF, 512, format, ##__VA_ARGS__ ); \
    fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",wrnk,__func__,__FILE__,__LINE__, \
    " ", BUF); smarties::Warnings::flushAll(); } } while(0)

} // end namespace Warnings

} // end namespace smarties
#endif
