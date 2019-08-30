# Dockerfile describing development environments and builds for the a HPC Optimization Development Environment (HODE)
#
# Author: D. de Vries <daniel.devries@darcorp.com>
#
# Parts of the code were copied from <https://github.com/FEniCS/dolfinx/blob/master/Dockerfile>, licensed under LGPL.
# Original Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>
#
# To build HODE image:
#
#    docker build --target hode .
#

ARG PETSC_VERSION=3.11.3
ARG PETSC4PY_VERSION=3.11.0

ARG MAKEFLAGS
ARG PETSC_OPTFLAGS="-02 -g"
ARG PETSC_DEBUGGING="yes"

FROM ubuntu:18.04 as base
LABEL maintainer="D. de Vries <daniel.devries@darcorop.com>"
LABEL description="Base image for HPC Optimization Development Environment (HODE)"

USER root
WORKDIR /tmp

ENV PYTHONIOENCODING "UTF-8"

# Install required packages available from apt-get
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
        cmake \
        g++ \
        gfortran \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-math-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-timer-dev \
        libeigen3-dev \
        libhdf5-mpich-dev \
        liblapack-dev \
        libmpich-dev \
        libopenblas-dev \
        mpich \
        ninja-build \
        pkg-config \
        python3-dev \
        python3-pip \
        python3-setuptools && \
    apt-get -y install \
        doxygen \
        git \
        graphviz \
        sudo \
        valgrind \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install numpy, scipy, and mpi4py
# Note: numpy and scipy where originally installed using apt-get, but they were older versions. This grabs the new ones.
RUN pip3 install --no-cache-dir numpy scipy mpi4py

WORKDIR /root

########################################

FROM base as dev-env
LABEL maintainer="D. de Vries <daniel.devries@darcorop.com>"
LABEL description="Development environment with working PETSc installation."

ARG PETSC_VERSION
ARG PETSC4PY_VERSION

ARG MAKEFLAGS
ARG PETSC_OPTFLAGS

WORKDIR /tmp

# Install PETSc and SLEPc with real types.
RUN apt-get -qq update && \
    apt-get -y install bison flex python && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p petsc-src && tar -xf petsc-${PETSC_VERSION}.tar.gz -C petsc-src --strip-components 1 && \
    cd petsc-src && \
    ./configure \
        --COPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --CXXOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --FOPTFLAGS=${PETSC_SLEPC_OPTFLAGS} \
        --with-debugging=${PETSC_SLEPC_DEBUGGING} \
        --with-fortran-bindings=no \
        --download-blacs \
        --download-hypre \
        --download-metis \
        --download-mumps \
        --download-ptscotch \
        --download-scalapack \
        --download-spai \
        --download-suitesparse \
        --download-superlu \
        --with-scalar-type=real \
        --prefix=/usr/local/petsc && \
    make ${MAKEFLAGS} && \
    make install && \
    export PETSC_DIR=/usr/local/petsc && \
    apt-get -y purge bison flex python && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV PETSC_DIR=/usr/local/petsc

# Install petsc4py
RUN pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION}

WORKDIR /root

########################################

FROM dev-env as hode
LABEL maintainer="D. de Vries <daniel.devries@darcorop.com>"
LABEL description="HPC Optimization Development Environment (HODE)"

WORKDIR /tmp

# Install OpenMDAO, Platupus, and psutil
RUN pip3 install --no-cache-dir openmdao platypus-opt psutil

# Install pyOptSparse
RUN apt-get -qq update && \
    apt-get -y install swig unzip && \
    wget -O pyoptsparse.zip https://github.com/mdolab/pyoptsparse/archive/master.zip && \
    unzip pyoptsparse.zip && \
    cd pyoptsparse-master && \
    python3 setup.py install && \
    apt-get -y purge swig unzip && \
    apt-get -y autoremove && \
    apt-get clean && \
    rm -rf /tmp/* && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /root
