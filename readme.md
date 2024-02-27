# Parallel matrix multiplication

***Fox's algorithm***


This repository contains some implementations of Fox's algorithm for parallel matriceis multiplication. 
You also can find here some utility functions for reading, printing, creating and comparing matricies. 

Use different test methods in `main.c` to compare perfomance of different implementations.

C language implementations use flat arrays of type `double*` and size $n^2$ to store square matricies of size $n \times n$. 

----

### TODO:
1. Erlang version;
2. Haskell version;
3. FPTL version (see FPTL [GitHub repo](https://github.com/Zumisha/FPTL));

----

### Installment

- I don't know how to run **OpenMP** library on Mac, so it's better to use Windows with Visual Studio. It should be avaliable by default.
  To actually use OpenMP go to your C++ *project properties* -> *C/C++* -> *language* -> *Open MP support*.
- **MPI** library may be not installed with Visual Studio, but you can get it from [microsoft.com](https://www.microsoft.com/en-us/download/details.aspx?id=105289). 
  Also MPI might not work, so you will have to correctly add references to its files, see this [tutorial](https://610yilingliu.github.io/2020/07/21/ConfigureOpenMPI/). 
  If this won't help, you might need download `msmpi.dll` from internet and place it nearby.


