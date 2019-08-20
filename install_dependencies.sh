#!/bin/bash

set -e

# https://stackoverflow.com/a/23378780/2203044
# LOGICAL_CORE_COUNT=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)
PHYSICAL_CORE_COUNT=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)

# Parameters modifiable from environment.
JOBS=${JOBS:-$PHYSICAL_CORE_COUNT}
SOURCES=${SOURCES:-$PWD/extern}
INSTALL_PATH=${INSTALL_PATH:-$PWD/extern/build}
CC=${CC:-gcc}
CXX=${CXX:-g++}
# Shorthands for versions.
# NOTE: Changing these numbers may not be enough for the script to work properly!

MPICH_VERSION=3.3

# Other shorthands.
TAR="tar --keep-newer-files"

# Flags. By default all are disabled.
INSTALL_MPICH=
INSTALL_OBLAS=
UNKNOWN_ARGUMENT=
PRINT_EXPORT=

# Determine what the user wants.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--all   ) INSTALL_MPICH=1   ;
                     INSTALL_OBLAS=1   ; shift ;;
        -e|--export) PRINT_EXPORT=1    ; shift ;;
           --mpich ) INSTALL_MPICH=1   ; shift ;;
           --blas  ) INSTALL_OBLAS=1   ; shift ;;
         *         ) UNKNOWN_ARGUMENT=1; shift ;;
    esac
done


if [ -z "$INSTALL_MPICH" -a -z "$INSTALL_OBLAS" -a -z "$PRINT_EXPORT" -o -n "$UNKNOWN_ARGUMENT" ]; then
    echo "Usage:
    ./install_dependencies [-a | --all | [[--mpich]] | [[--blas]] ]

Arguments:
  -a,  --all    - Install all available libraries and tools
  -e,  --export - Print export commands for all libraries and tools
                  (assuming they are installed)
  --mpich       - Install mpich version ${MPICH_VERSION}
  --blas        - Install serial OpenBLAS

All libraries and tools are installed locally in the extern/ folder.
Note that this script tries not to redo everything from scratch if run multiple
times. Therefore, in case of errors, try erasing the relevant subfolders inside extern/.

"
    exit
fi


BASEPWD=$PWD


if [ -n "$INSTALL_MPICH" ]; then
    echo "Installing mpich ${MPICH_VERSION}..."
    wget -nc http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz -P $SOURCES
    cd $SOURCES
#    $TAR -xzvf mpich-${MPICH_VERSION}.tar.gz
    cd mpich-${MPICH_VERSION}
    CC=${CC} CXX=${CXX} ./configure \
      --prefix=$INSTALL_PATH/mpich-${MPICH_VERSION}/ \
      --enable-fast=all --enable-fortran=no --enable-threads=multiple
    make -j${JOBS}
    make install -j${JOBS}
    cd $BASEPWD
fi

if [ -n "$INSTALL_OBLAS" ]; then
    echo "Installing OpenBLAS..."
    cd $SOURCES
    git clone https://github.com/xianyi/OpenBLAS
    cd OpenBLAS
    make CC=${CC} FC=${CC} NUM_THREADS=1 USE_THREAD=0 USE_OPENMP=0 -j${JOBS}
    make PREFIX=$INSTALL_PATH/OpenBLAS/ install
    cd $BASEPWD
fi

#if [ -n "$INSTALL_MPICH" ]; then
#    echo
#    echo "======================================================================"
#    echo "Done! Run or add to ~/.bashrc the following command:"
#    echo
#fi
#if [ -n "$INSTALL_MPICH" -o -n "$PRINT_EXPORT" ]; then
#    echo "export PATH=$INSTALL_PATH/mpich-${MPICH_VERSION}/bin:\$PATH"
#fi
