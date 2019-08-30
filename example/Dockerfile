# The HODE image either has to be build first, or it can be pulled from Docker Hub
FROM danieldv/hode:latest

# Add example.py script and switch to /root dir
ADD example.py /root
WORKDIR /root

# The user can specify the number of processors, np, Rosenbrock dimensionality, dim,
# and number of bits to encode components of x using environment variables.
ENV np=1 dim=2 bits=31
CMD mpirun -np $np python3 example.py $dim $bits
