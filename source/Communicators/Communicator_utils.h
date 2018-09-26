//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Panagiotis Hadjidoukas.
//

#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <dirent.h>

#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <limits>

inline void intToDoublePtr(const int i, double*const ptr)
{
  assert(i>=0);
  *ptr = (double)i+0.1;
}
inline int doublePtrToInt(const double*const ptr)
{
  return (int)*ptr;
}
inline double* _alloc(const int size)
{
    double* ret = (double*) malloc(size);
    memset(ret, 0, size);
    return ret;
}
inline void _dealloc(double* ptr)
{
  if(ptr not_eq nullptr) {
    free(ptr);
    ptr=nullptr;
  }
}

inline void launch_exec(const std::string exec, const int socket_id)
{
  printf("About to exec %s.... \n",exec.c_str());
  const int res = execlp(exec.c_str(),
      exec.c_str(),
      std::to_string(socket_id).c_str(),
      NULL);
  //int res = execvp(*largv, largv);
  if (res < 0)
    fprintf(stderr,"Unable to exec file '%s'!\n", exec.c_str());
}

inline int redirect_stdout_stderr()
{
  fflush(0);
  char output[256];
  sprintf(output, "output");
  int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
  dup2(fd, 1);    // make stdout go to file
  dup2(fd, 2);    // make stderr go to file
  close(fd);      // fd no longer needed
  return fd;
}

int recv_all(int fd, void *buffer, unsigned int size);

int send_all(int fd, void *buffer, unsigned int size);

int parse2(char *line, char **argv);

int cp(const char *from, const char *to);

int copy_from_dir(const std::string name);

void comm_sock(int fd, const bool bsend, double*const data, const int size);

#ifdef MPI_VERSION
inline int getRank(const MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
inline int getSize(const MPI_Comm comm)
{
  int size;
  MPI_Comm_size(comm, &size);
  return size;
}
#endif
