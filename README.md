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
Sparse Rows (CSR) layouts.

Usage
----
**Load a MAT file (MATLAB(R) binary format) into memory:**
```cuda
SparseMatrix sp("sample.mat");
```
Since MAT files store nonzero entries in CSC layout, `SparseMatrix` will do so
as well. All the work is done in its constructor.

**Convert the matrix into CSC layout:**
```cuda
SparseMatrixCSR sp_csr(sp);
```
Again, the constructor does all the heavy lifting of format conversion.

**Copy a matrix (CSC layout) to GPU memory:**
```cuda
SparseMatrix sp("sample.mat"); // load a sparse matrix to host memory
SparseMatrixDevice spd(sp); // copy the inner data to GPU ...
// ... and then copy the container itself to GPU
SparseMatrixDevice* spdp;
cudaMalloc(&spdp, sizeof(SparseMatrixDevice));
cudaMemcpy(spdp, &spd, sizeof(SparseMatrixDevice), cudaMemcpyHostToDevice);
```
AWe still have to remeber to copy both the structure and its member arrays, but
the use of constructor helps keep the code short.

**Copy a matrix (CSR layout) to GPU memory:**
```cuda
SparseMatrix sp("sample.mat"); // load a sparse matrix to host memory
SparseMatrixCSR sp_csr(sp); // convert the matrix into CSR layout
SparseMatrixDeviceCSR spd(sp_csr); // copy the inner data to GPU ...
// ... and then copy the container itself to GPU
SparseMatrixDeviceCSR* spdp;
cudaMalloc(&spdp, sizeof(SparseMatrixDeviceCSR));
cudaMemcpy(spdp, &spd, sizeof(SparseMatrixDeviceCSR), cudaMemcpyHostToDevice);
```

**Pass a sparse matrix into a host function:**
```cpp
void serial_dotprod(const SparseMatrix& sp, int col1, int col2)
{
    ...
}
```
IMPORTANT: Pass the object by reference, not by value. The containers support
neither copy constructors nor assignments; we felt that copy semantics are too
costly to leave to implicit behaviors. If you really want to make a copy of
`SparseMatrix`, simply do a memcpy for each data member.

**Pass a sparse matrix into a CUDA kernel:**

First copy the matrix to GPU memory; then just pass the pointer into the kernel.
```cuda
SparseMatrix sp("sample.mat"); // load a sparse matrix to host memory
SparseMatrixCSR sp_csr(sp); // convert the matrix into CSR layout
SparseMatrixDeviceCSR spd(sp_csr); // copy the inner data to GPU ...
// ... and then copy the container itself to GPU
SparseMatrixDeviceCSR* spdp;
cudaMalloc(&spdp, sizeof(SparseMatrixDeviceCSR));
cudaMemcpy(spdp, &spd, sizeof(SparseMatrixDeviceCSR), cudaMemcpyHostToDevice);
dotprod<<<1024, 1024>>>(spdp);
```

How to compile the library
----
```bash
make
make tester
```

How to compile programs that use the library
----
```bash
nvcc -O3 -arch=sm_35 -rdc=true -o tester tester.cu -L[path to libspmat.a]
-lspmat -lz -lm
```
Do not forget `-lz -lm` at the end of the compilation option.

Dependencies
----
  - MAT File I/O Libary: http://sourceforge.net/projects/matio/
  
    I've included a script named `get_matio.sh` that attempts to download and
install this dependency. *In the event where the script fails:* download the
latest tarball from the project website and extract it to the `matio/`
subdirectory. Make sure to add `--prefix=$PWD/matio` to the run of
`./configure`, so that the compiled result will stay inside of the current
directory.

Credits
----
Hyunsu "Philip" Cho and Sam Johnson, Trinity College, Hartford, CT
