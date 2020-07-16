//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_SocketsLib_h
#define smarties_SocketsLib_h

#include <unistd.h> // usleep
#include <sys/un.h> // sockaddr_un
#include <sys/socket.h> // send, recv, ...
#include <cstdio> // printf
#include <cstring> // strcpy, strlen, bzero
#include <cassert>
#include <vector>

namespace smarties
{

enum SOCKET_MSG_TYPE {SOCKET_SEND, SOCKET_RECV};
struct SOCKET_REQ
{
  int client;
  void * buffer;
  unsigned size;
  unsigned todo;
  SOCKET_MSG_TYPE type;
};

////////////////////////////////////////////////////////////////////////////////

inline int SOCKET_Test(int& completed, SOCKET_REQ& req)
{
  if(req.todo == 0) { completed = 1; return 0; }
  const unsigned transferred_size = req.size - req.todo;
  char* const todo_buffer = (char*) req.buffer + transferred_size;
  assert(transferred_size <= req.size);
  const int bytes = req.type == SOCKET_SEND
                  ? send(req.client, todo_buffer, req.todo, 0)
                  : recv(req.client, todo_buffer, req.todo, 0);
  if(bytes > 0) {
    assert((unsigned) bytes <= req.todo);
    req.todo -= bytes;
    completed = req.todo == 0;
    return 0;
  } else {
    printf("FATAL: lost connection in communication over sockets\n");
    fflush(0); abort(); return -1;
  }
}

inline int SOCKET_Wait(SOCKET_REQ& req)
{
  if(req.todo == 0) return 0;
  const unsigned transferred_size = req.size - req.todo;
  char* todo_buffer = (char*) req.buffer + transferred_size;
  assert(transferred_size <= req.size);
  while (req.todo > 0) {
    const int bytes = req.type == SOCKET_SEND
                    ? send(req.client, todo_buffer, req.todo, 0)
                    : recv(req.client, todo_buffer, req.todo, 0);
    if( bytes > 0 ) {
      assert((unsigned) bytes <= req.todo);
      req.todo -= bytes;
      todo_buffer += bytes;
    } else {
      printf("FATAL: lost connection in communication over sockets\n");
      fflush(0); abort(); return -1;
    }
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////


inline int SOCKET_Irecv(void* const buffer,
                        const unsigned size,
                        const int socketid,
                        SOCKET_REQ& request
) {
  request.client = socketid; request.buffer = buffer;
  request.size = size; request.todo = size; request.type = SOCKET_RECV;
  //return SOCKET_Test(request.completed, request);
  return 0;
}

inline int SOCKET_Isend(void* const buffer,
                        const unsigned size,
                        const int socketid,
                        SOCKET_REQ& request
) {
  request.client = socketid; request.buffer = buffer;
  request.size = size; request.todo = size; request.type = SOCKET_SEND;
  //return SOCKET_Test(request.completed, request);
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

inline int SOCKET_Brecv(void* const buffer,
                        const unsigned size,
                        const int socketid)
{
  unsigned bytes_to_receive = size;
  char* pos = (char*) buffer;
  while (bytes_to_receive > 0) {
    const int bytes = recv(socketid, pos, bytes_to_receive, 0);
    //printf("recv %d bytes out of %d\n", bytes, bytes_to_receive);
    if( bytes > 0 ) {
      assert((unsigned) bytes <= bytes_to_receive);
      bytes_to_receive -= bytes;
      pos += bytes;
    } else {
      printf("FATAL: lost connection in communication over sockets\n");
      fflush(0); abort(); return -1;
    }
  }
  return 0;
}


inline int SOCKET_Bsend(const void* const buffer,
                        const unsigned size,
                        const int socketid)
{
  unsigned bytes_to_send = size;
  const char* pos = (const char*) buffer;
  while ( bytes_to_send > 0 ) {
    const int bytes = send(socketid, pos, bytes_to_send, 0);
    //printf("sent %d bytes out of %d\n", bytes, bytes_to_send);
    if( bytes > 0 ) {
      assert((unsigned) bytes <= bytes_to_send);
      bytes_to_send -= bytes;
      pos += bytes;
    } else {
      printf("FATAL: lost connection in communication over sockets\n");
      fflush(0); abort(); return -1;
    }
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

inline int SOCKET_clientConnect()
{
  // Specify the socket
  char SOCK_PATH[] = "smarties_AFUNIX_socket_FD";

  int serverAddr;
  if( ( serverAddr = socket(AF_UNIX, SOCK_STREAM, 0) ) == -1 )
  {
    printf("SOCKET_clientConnect::socket failed"); fflush(0); abort();
  }

  {
    int _TRU = 1;
    if(setsockopt(serverAddr, SOL_SOCKET, SO_REUSEADDR, &_TRU, sizeof(int))<0)
    {
      printf("SOCKET_clientConnect::setsockopt failed\n"); fflush(0); abort();
    }
  }

  // Specify the server
  struct sockaddr_un serverAddress;
  bzero((char *)&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path)+1;

  // Connect to the server
  size_t nAttempts = 0;
  while (connect(serverAddr, (struct sockaddr *)&serverAddress, servlen) < 0)
  {
    if(++nAttempts % 1000 == 0) {
      printf("Application is taking too much time to connect to smarties."
             " If your application needs to change directories (e.g. set up a"
             " dedicated directory for each run) it should do so AFTER"
             " the connection to smarties has been initialzed.\n");
    }
    usleep(1);
  }

  return serverAddr;
}

inline int SOCKET_serverConnect(const unsigned nClients,
                             std::vector<int>& clientSockets)
{
  // Specify the socket
  char SOCK_PATH[] = "smarties_AFUNIX_socket_FD";
  unlink(SOCK_PATH);

  int serverAddr;
  if ( ( serverAddr = socket(AF_UNIX, SOCK_STREAM, 0) ) == -1 ) {
    printf("SOCKET_serverConnect::socket failed"); fflush(0); abort();
  }

  struct sockaddr_un serverAddress;
  bzero(&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  //this printf is to check that there is no funny business with trailing 0s:
  //printf("%s %s\n",serverAddress.sun_path, SOCK_PATH); fflush(0);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path) +1;

  if ( bind(serverAddr, (struct sockaddr *)&serverAddress, servlen) < 0 ) {
    printf("SOCKET_serverConnect::bind failed"); fflush(0); abort();
  }

  if (listen(serverAddr, nClients) == -1) { // liste
    printf("SOCKET_serverConnect::listen failed"); fflush(0); abort();
  }

  clientSockets.resize(nClients, 0);
  for(unsigned i = 0; i<nClients; ++i)
  {
    struct sockaddr_un clientAddress;
    unsigned int addr_len = sizeof(clientAddress);
    struct sockaddr*const addrPtr = (struct sockaddr*) &clientAddress;
    if( ( clientSockets[i] = accept(serverAddr, addrPtr, &addr_len) ) == -1 )
    {
      printf("SOCKET_serverConnect::accept failed"); fflush(0); abort();
    }
    printf("server: new connection on socket %d\n", clientSockets[i]);
  }

  return serverAddr;
}

}
#endif
