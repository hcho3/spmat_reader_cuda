#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "matio.h"
#include "spmat.h"

SparseMatrix::SparseMatrix(const char *filename)
{
    mat_t *matfp;
    matvar_t *matvar;
    mat_sparse_t *A;

    matfp = Mat_Open(filename, MAT_ACC_RDONLY);

    if ( NULL == matfp ) {
        fprintf(stderr, "Error opening MAT file \"%s\"!\n", filename);
        exit(1);
    }
    matvar = Mat_VarRead(matfp, "A");
    if ( NULL == matvar ) {
        fprintf(stderr,"Variable 'A' not found, or error "
                "reading MAT file\n");
        Mat_Close(matfp);
        exit(1);
    }

    nrows = matvar->dims[0];
    ncols = matvar->dims[1];

    A = (mat_sparse_t *)matvar->data;

    nnz = A->ndata;

    cscVal = (double *)malloc(nnz * sizeof(double));
    memcpy(cscVal, A->data, nnz * sizeof(double)); // nonzero entries

    cscRowInd = (int *)malloc(nnz * sizeof(int));
    memcpy(cscRowInd, A->ir, nnz * sizeof(int));

    cscColPtr = (int *)malloc(A->njc * sizeof(int));
    memcpy(cscColPtr, A->jc, A->njc * sizeof(int));

    Mat_VarFree(matvar);
    Mat_Close(matfp);
}

SparseMatrix::~SparseMatrix()
{
    free(cscVal);
    free(cscRowInd);
    free(cscColPtr);
}

SparseMatrixDevice::SparseMatrixDevice(const SparseMatrix& sp)
{
    nnz = sp.nnz;
    nrows = sp.nrows;
    ncols = sp.ncols;
    cudaMalloc(&cscVal, nnz * sizeof(double));
    cudaMalloc(&cscRowInd, nnz * sizeof(int));
    cudaMalloc(&cscColPtr, (ncols+1) * sizeof(int));

    cudaMemcpy(cscVal, sp.cscVal, nnz * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(cscRowInd, sp.cscRowInd, nnz * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(cscColPtr, sp.cscColPtr, (ncols+1)*sizeof(int),
        cudaMemcpyHostToDevice);
}

SparseMatrixDevice::~SparseMatrixDevice()
{
    cudaFree(cscVal);
    cudaFree(cscRowInd);
    cudaFree(cscColPtr);
}

SparseMatrixCSR::SparseMatrixCSR(const SparseMatrix& sp)
{
    nnz = sp.nnz;
    nrows = sp.nrows;
    ncols = sp.ncols;
    csrVal = (double *)malloc(nnz * sizeof(double)); // same # of nonzeros
    csrColInd = (int *)malloc(nnz * sizeof(int));
    csrRowPtr = (int *)malloc((nrows+1) * sizeof(int));

    int row, col, i;
    double val;

    // create as many buckets as the rows in the matrix
    //     buckets is an array of vectors
    //     vector<int> is a dynamic-sized array of int's
    //     use "new" instead of "malloc" to invoke vector's constructor
    std::vector<int> *buckets = new std::vector<int>[nrows];

    // create another set of buckets to hold the values
    std::vector<double> *buckets_val = new std::vector<double>[nrows];

    // Inspect one column at a time, and append the column index at the end of
    // appropriate buckets. In the end, each i-th bucket should have the list
    // of all columns that have nonzeros at the i-th row.

    for (col = 0; col < ncols; col++) {
        // traverse through all the nonzero elements in column col.
        for (i = sp.cscColPtr[col]; i < sp.cscColPtr[col+1]; i++) {
            // fetch the row index and value of each nonzero element
            row = sp.cscRowInd[i];
            val = sp.cscVal[i];

            // append the column index and the element value to right bucket
            buckets[row].push_back(col);
            buckets_val[row].push_back(val);
        }
    }

    // flatten the buckets into CSR format
    // since vector is a C++ object, use std::copy instead of memcpy
    csrRowPtr[0] = 0;
    for (row = 0; row < nrows; row++) {
        std::copy(buckets[row].begin(), buckets[row].end(),
                &csrColInd[ csrRowPtr[row] ]);
        std::copy(buckets_val[row].begin(), buckets_val[row].end(),
                &csrVal[ csrRowPtr[row] ]);

        // store the range for this row
        // we now know the position where we paste the next bucket into
        csrRowPtr[row+1] = csrRowPtr[row] + buckets[row].size();
    }

    // free up memory
    //     use "delete" instead of "free" to invoke vector's destructor
    delete [] buckets;
    delete [] buckets_val;
}

SparseMatrixCSR::~SparseMatrixCSR()
{
    free(csrVal);
    free(csrColInd);
    free(csrRowPtr);
}

SparseMatrixDeviceCSR::SparseMatrixDeviceCSR(const SparseMatrixCSR& sp)
{
    nnz = sp.nnz;
    nrows = sp.nrows;
    ncols = sp.ncols;
    cudaMalloc(&csrVal, nnz * sizeof(double));
    cudaMalloc(&csrColInd, nnz * sizeof(int));
    cudaMalloc(&csrRowPtr, (nrows+1) * sizeof(int));

    cudaMemcpy(csrVal, sp.csrVal, nnz * sizeof(double),
        cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, sp.csrColInd, nnz * sizeof(int),
        cudaMemcpyHostToDevice);
    cudaMemcpy(csrRowPtr, sp.csrRowPtr, (nrows+1)*sizeof(int),
        cudaMemcpyHostToDevice);
}

SparseMatrixDeviceCSR::~SparseMatrixDeviceCSR()
{
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);
}
