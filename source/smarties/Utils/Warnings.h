//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Warnings_h
#define smarties_Warnings_h

namespace smarties
{
namespace Warnings
{
enum InfoDumpLevel { SILENT = 0, WARNINGS, DEBUG, SCHEDULER, LEARNERS };

#ifdef NDEBUG
static constexpr InfoDumpLevel level = WARNINGS;
#else
static constexpr InfoDumpLevel level = DEBUG;
//static constexpr InfoDumpLevel level = LEARNERS;
//static constexpr InfoDumpLevel level = SCHEDULER;
#endif

void signal_handler [[ noreturn ]] (int signal);

void init_warnings();

void print_warning(const char * funcname, const char * filename,
                   int line, const char * fmt, ...);
void print_stacktrace();

void mpi_abort();

#define    die(err_message)      do {                                          \
  using namespace smarties::Warnings;                                          \
  print_warning(__func__, __FILE__, __LINE__, err_message);                    \
  print_stacktrace(); mpi_abort(); } while(0)

#define   _die(format, ...) do {                                               \
  using namespace smarties::Warnings;                                          \
  print_warning(__func__, __FILE__, __LINE__, format, __VA_ARGS__);            \
  print_stacktrace(); mpi_abort(); } while(0)

#define   warn(err_message)  do { \
  if(Warnings::level >= smarties::Warnings::WARNINGS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, err_message);                  \
  } } while(0)

#define  _warn(format, ...)  do { \
  if(Warnings::level >= smarties::Warnings::WARNINGS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, __VA_ARGS__);          \
  } } while(0)

#define debugS(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::SCHEDULER) {                       \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, __VA_ARGS__);          \
  } } while(0)

#define _debugL(format, ...)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, __VA_ARGS__);          \
  } } while(0)

#define debugL(err_message)  do { \
  if(Warnings::level == smarties::Warnings::LEARNERS) {                        \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, err_message);                  \
  } } while(0)

#define _debug(format, ...)  do { \
  if(Warnings::level >= smarties::Warnings::DEBUG) {                           \
    using namespace smarties::Warnings;                                        \
    print_warning(__func__, __FILE__, __LINE__, format, __VA_ARGS__);          \
  } } while(0)

} // end namespace Warnings

} // end namespace smarties
#endif
