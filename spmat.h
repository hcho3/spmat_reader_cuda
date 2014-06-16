class Nocopy { /* prohibit object copying for all subclasses */
protected:
    Nocopy() {}
    ~Nocopy() {}
private:
    Nocopy(const Nocopy &);
    const Nocopy& operator=(const Nocopy &);
};

struct SparseMatrix : Nocopy { // CSC layout, main memory
    int nnz;
    int nrows;
    int ncols;
    double *cscVal;
    int *cscRowInd;
    int *cscColPtr;

    SparseMatrix(const char *filename); // read MATLAB(R) file
    ~SparseMatrix();
};

struct SparseMatrixDevice : Nocopy { // CSC layout, GPU memory
    int nnz;
    int nrows;
    int ncols;
    double *cscVal;
    int *cscRowInd;
    int *cscColPtr;

    SparseMatrixDevice *devptr; // use this pointer for kernel launches

    // load matrix from main memory
    SparseMatrixDevice(const SparseMatrix& sp);
    ~SparseMatrixDevice();
};

struct SparseMatrixCSR : Nocopy { // CSR layout, main memory
    int nnz;
    int nrows;
    int ncols;
    double *csrVal;
    int *csrColInd;
    int *csrRowPtr;

    SparseMatrixCSR(const SparseMatrix& sp); // convert CSC to CSR
    ~SparseMatrixCSR();
};

struct SparseMatrixDeviceCSR : Nocopy {
    int nnz;
    int nrows;
    int ncols;
    double *csrVal;
    int *csrColInd;
    int *csrRowPtr;

    SparseMatrixDeviceCSR *devptr; // use this pointer for kernel launches

    // load matrix from main memory
    SparseMatrixDeviceCSR(const SparseMatrixCSR& sp);
    ~SparseMatrixDeviceCSR();
};
