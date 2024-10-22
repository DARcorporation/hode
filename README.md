# HPC Optimization Development Environment (HODE)

This repo contains the Dockerfile for the HODE image. It is a development environment which can be deployed to HPC 
environments to perform optimization tasks using [OpenMDAO](https://openmdao.org/), 
[pyOptSparse](https://github.com/mdolab/pyoptsparse), and
[Platypus](https://github.com/Project-Platypus/Platypus).
 
The Docker container contains working installations of MPI (mpich) and PETSc. 
 
OpenMDAO, pyOptSparse, and Platypus are installed for Python 3.
 
 There is an [example](./example) usage of the HODE image included in this repo.

# Credits
Development of this container was performed at [DARcorporation](https://www.darcorp.com/).
