#include <stdio.h>
#include "spmat.h"

int main(void)
{
    SparseMatrix sp("sample.mat");
    SparseMatrixCSR sp_csr(sp);

    int i;

    printf("See http://docs.nvidia.com/cuda/cusparse/index.html"
           "#compressed-sparse-row-format-csr to see the example matrix.\n");

    printf("CSC format:\n");
    printf("  nnz = %d\n", sp.nnz);
    printf("  nrows = %d\n", sp.nrows);
    printf("  ncols = %d\n", sp.ncols);

    printf("  cscVal =\n    ");
    for (i = 0; i < sp.nnz; i++)
        printf("%.1lf ", sp.cscVal[i]);
    printf("\n");
    printf("  cscRowInd =\n    ");
    for (i = 0; i < sp.nnz; i++)
        printf("%-3d ", sp.cscRowInd[i]);
    printf("\n");
    printf("  cscColPtr =\n    ");
    for (i = 0; i <= sp.ncols; i++)
        printf("%-3d ", sp.cscColPtr[i]);
    printf("\n");

    printf("CSR format:\n");
    printf("  nnz = %d\n", sp_csr.nnz);
    printf("  nrows = %d\n", sp_csr.nrows);
    printf("  ncols = %d\n", sp_csr.ncols);

    printf("  csrVal =\n    ");
    for (i = 0; i < sp_csr.nnz; i++)
        printf("%.1lf ", sp_csr.csrVal[i]);
    printf("\n");
    printf("  csrColInd =\n    ");
    for (i = 0; i < sp_csr.nnz; i++)
        printf("%-3d ", sp_csr.csrColInd[i]);
    printf("\n");
    printf("  csrRowPtr =\n    ");
    for (i = 0; i <= sp_csr.nrows; i++)
        printf("%-3d ", sp_csr.csrRowPtr[i]);
    printf("\n");

    return 0;
}
