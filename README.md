**`spmat_reader_cuda`: minimalistic, CUDA-aware C++ container for
sparse matrices**

This library provides several C++ containers:
  - `SparseMatrix`: uses CSC layout; stores data on host memory
  - `SparseMatrixCSR`: uses CSR layout; store data on host memory
  - `SparseMatrixDevice`: uses CSC layout; stores data on device memory
  - `SparseMatrixDeviceCSR`: uses CSR layout; stores data on device memory

The containers are nothing more than a collection of three arrays that together
indicate the values and positions of nonzero entries. See
http://docs.nvidia.com/cuda/cusparse/#matrix-formats
for a detailed explanation for Compressed Sparse Columns (CSC) and Compressed
Sparse Rows (CSR) layouts. (Use Firefox to properly display equations in the
documentation.)

Usage
----
See `tester.cu` for a short sample program.
**Load a MAT file (MATLAB(R) binary format) into memory:**
```cuda
#include "spmat.h"
...
SparseMatrix sp("sample.mat");
```
Since MAT files store nonzero entries in CSC layout, `SparseMatrix` will do so
as well. All the work is done in its constructor.

**Convert the matrix into CSC layout:**
```cuda
#include "spmat.h"
...
SparseMatrixCSR sp_csr(sp);
```
Again, the constructor does all the heavy lifting of format conversion.

**Copy a matrix (CSC layout) to GPU memory:**
```cuda
#include "spmat.h"
...
SparseMatrix sp("sample.mat");
SparseMatrixDevice spd(sp);

kernel<<<1, 1>>>(spd.devptr);
...
__global__ void kernel(SparseMatrixDevice *spd)
{
    // use spd->cscVal, spd->cscRowInd, and spd->cscColPtr
}
```
Just remember to use `devptr` member when calling a GPU kernel.

**Copy a matrix (CSR layout) to GPU memory:**
```cuda
#include "spmat.h"
...
SparseMatrix sp("sample.mat");
SparseMatrixCSR sp_csr(sp); // convert the matrix into CSR layout
SparseMatrixDeviceCSR spd_csr(sp_csr);

kernel<<<1, 1>>>(spd_csr.devptr);
...
__global__ void kernel(SparseMatrixDeviceCSR *spd)
{
    // use spd->csrVal, spd->csrColInd, spd->csrRowPtr
}
```
Here, too, remember to use the `devptr` member.

**Pass a sparse matrix into a host function:**
```cpp
void serial_dotprod(const SparseMatrix& sp, int col1, int col2)
{    
    // use sp.cscVal, sp.cscRowInd, sp.cscColPtr
}
```
IMPORTANT: Pass the object by reference, not by value. The containers support
neither copy constructors nor assignments; we felt that copy semantics are too
costly to leave to implicit behaviors. If you really want to make a copy of
`SparseMatrix`, simply do a memcpy for each data member.

**De-allocate a sparse matrix**
Thanks to the destructor, `SparseMatrix` and `SparseMatrixDevice` will be
de-allocated whenever they go outside of scope.

How to compile the library
----
```bash
make
make tester
```

How to compile programs that use the library
----
```bash
nvcc -O3 -arch=sm_35 -rdc=true -o tester tester.cu -I[path to spmat.h]
-L[path to libspmat.a] -lspmat -lz -lm
```
Do not forget `-lz -lm` at the end of the compilation option.

Dependencies
----
  - MAT File I/O Library: http://sourceforge.net/projects/matio/
  
    I've included a script named `get_matio.sh` that attempts to download and
install this dependency. *In the event where the script fails:* download the
latest tarball from the project website and extract it to the `matio/`
subdirectory. Make sure to add `--prefix=$PWD/matio` to the run of
`./configure`, so that the compiled result will stay inside of the current
directory.

Credits
----
Hyunsu "Philip" Cho and Sam Johnson, Trinity College, Hartford, CT
