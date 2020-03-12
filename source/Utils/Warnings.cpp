//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Warnings.h"

//#define PRINT_STACK_TRACE

#ifdef PRINT_STACK_TRACE
#define BACKWARD_HAS_DW 0
#define BACKWARD_HAS_BFD 0
#define BACKWARD_HAS_DWARF 0
#define BACKWARD_HAS_UNWIND 0
#define BACKWARD_HAS_BACKTRACE 0
#define BACKWARD_HAS_BACKTRACE_SYMBOL 0
#include "../extern/backward.hpp"
#endif

#include <mutex>
//#include <sstream>
#include <stdarg.h>
#include <csignal>
#include <iostream>

#include "MPIUtilities.h"

namespace smarties
{
namespace Warnings
{
static std::mutex warn_mutex;

void signal_handler [[ noreturn ]] (int signal)
{
  if (signal == SIGABRT) {
      std::cerr << "SIGABRT received\n";
  } else {
      std::cerr << "Unexpected signal " << signal << " received\n";
  }
  print_stacktrace();
  std::_Exit(EXIT_FAILURE);
}

void init_warnings()
{
  #ifdef PRINT_STACK_TRACE
  static backward::SignalHandling sh;
  #endif

  // Setup handler
  auto previous_handler = std::signal(SIGABRT, signal_handler);
  if (previous_handler == SIG_ERR) die("Error setup failed");
}

void print_warning(const char * funcname, const char * filename,
                   int line, const char * fmt, ...)
{
  std::lock_guard<std::mutex> wlock(smarties::Warnings::warn_mutex);
  const auto wrnk = smarties::MPIworldRank();

  char BUF[512];
  va_list args;
  va_start (args, fmt);
  vsnprintf (BUF, 512, fmt, args);
  va_end (args);

  fprintf(stderr,"Rank %d %s(%s:%d)%s %s\n",
    wrnk, funcname, filename, line, " ", BUF);
  fflush(stdout); fflush(stderr); fflush(0);
}

void print_stacktrace()
{
  #ifdef PRINT_STACK_TRACE
    std::ostringstream strace;
    using namespace backward;

    StackTrace st;
    st.load_here(100);
    Printer p;
    p.object = true;
    p.color_mode = ColorMode::automatic;
    p.address = true;
    p.print(st, strace);

    fwrite(strace.str().c_str(), sizeof(char), strace.str().size(), stderr);
    fflush(stdout); fflush(stderr); fflush(0);
  #endif
}

void mpi_abort()
{
  MPI_Abort(MPI_COMM_WORLD, 1);
}

} // end namespace Warnings

} // end namespace smarties

