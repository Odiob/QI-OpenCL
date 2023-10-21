# QI-OpenCL
Final project for Quantum Information course at the University of Padua (2019). 

The code implements the shifted [QR algorithm](https://en.wikipedia.org/wiki/QR_algorithm) to find the lowest eigenvalue of a random hermitian matrix.

Made for learning the basics of parallel computing and OpenCL.

Tested with the following configuration:
- CPU: Intel Core i5 4670
- GPU: AMD Radeon HD 7870
- OS: Windows 10
- Compiler: GCC 8.1.0 via [MinGW-w64](https://www.mingw-w64.org/)
- External libraries: [AMD APP SDK](https://en.wikipedia.org/wiki/AMD_APP_SDK), [LAPACKE](https://www.netlib.org/lapack/lapacke.html)

Note: code was never intended to run on different hardware and/or software, hence it might require some modifications to do it.
