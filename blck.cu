#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverMg.h>

int main() {
    int rows = 4; // Assume even number of rows and columns
    int cols = 4; // Assume even number of rows and columns
    int size_partition = (rows / 2) * (cols / 2);
    int halfRows = rows / 2;
    int halfCols = cols / 2;
    int size = rows * cols;

    // Allocate and initialize the 1D representation of the 2D matrix
    float *h_matrix = (float *)malloc(size * sizeof(float));
    if (h_matrix == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return EXIT_FAILURE;
    } // if
    
    // Fill the matrix with values from 1 to 16 (for a 4x4 matrix)
    for (int i = 0; i < size; ++i) {
        h_matrix[i] = (float)(i + 1);
    } // for


    float *h_A, *h_B, *h_C, *h_D;
    // malloc memory
    h_A = (float*) malloc( sizeof(float)*size_partition );
    h_B = (float*) malloc( sizeof(float)*size_partition );
    h_C = (float*) malloc( sizeof(float)*size_partition );
    h_D = (float*) malloc( sizeof(float)*size_partition );
    
    // Check allocation
    if (!(h_A && h_B && h_C && h_D && h_matrix)) {
        fprintf(stderr, "Memory allocation failed!\n");
        // Remember to free any previously allocated blocks here before exiting
        return EXIT_FAILURE;
    } // if

    // assigning the matrix into host variable
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j; // Index in the original matrix
            int partitionIndex = (i % halfRows) * halfCols + (j % halfCols); // Index in the partition
            if (i < halfRows) {
                if (j < halfCols) {
                    h_A[partitionIndex] = h_matrix[index];
                } // if
                else {
                    h_B[partitionIndex] = h_matrix[index];
                } // else
            } // if 
            else {
                if (j < halfCols) {
                    h_C[partitionIndex] = h_matrix[index];
                } // if
                else {
                    h_D[partitionIndex] = h_matrix[index];
                } // else
            } // else
        } // for
    } // for


    
    // malloc device variables
    float *d_A, *d_B, *d_C, *d_D, *d_Q;
    cudaMalloc(&d_A, sizeof(float) * size_partition);
    cudaMalloc(&d_B, sizeof(float) * size_partition);
    cudaMalloc(&d_C, sizeof(float) * size_partition);
    cudaMalloc(&d_D, sizeof(float) * size_partition);
    cudaMalloc(&d_Q, sizeof(float) * size_partition);

    // memcpy from host to device with corrected sizes
    cudaMemcpy(d_A, h_A, sizeof(float) * size_partition, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * size_partition, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * size_partition, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, sizeof(float) * size_partition, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolver_handle;
    cusolverDnCreate(&cusolver_handle);

    // Define alpha and beta for the SGEMM operations
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *d_temp;
    cudaMalloc(&d_temp, sizeof(float) * size_partition);

    int *d_pivot, *d_info;
    cudaMalloc((void**)&d_pivot, halfRows * sizeof(int));
    cudaMalloc((void**)&d_info, sizeof(int));


    // Get buffer size for LU decomposition
    int Lwork = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle, halfRows, halfRows, d_A, halfRows, &Lwork);

    // Allocate workspace for LU decomposition
    float* d_Workspace;
    cudaMalloc(&d_Workspace, sizeof(float) * Lwork);

    // Perform LU decomposition
    int* d_Pivot, *d_Info;
    cudaMalloc(&d_Pivot, sizeof(int) * halfRows);
    cudaMalloc(&d_Info, sizeof(int));
    cusolverDnSgetrf(cusolver_handle, halfRows, halfRows, d_A, halfRows, d_Workspace, d_Pivot, d_Info);


    // Allocate memory for the identity matrix on the host
    float *h_identity = (float *)malloc(halfRows * halfRows * sizeof(float));

    // Check for successful host allocation
    if (h_identity == NULL) {
        fprintf(stderr, "Host memory allocation for identity matrix failed!\n");
        return EXIT_FAILURE;
    } // if

    // Initialize the identity matrix on the host correctly
    for (int i = 0; i < halfRows * halfRows; ++i) {
        h_identity[i] = 0.0f;
    } // for
    for (int i = 0; i < halfRows; ++i) {
        h_identity[i * halfRows + i] = 1.0f;
    } // for

    // Allocate memory for the identity matrix on the device
    float *d_identity;
    cudaMalloc(&d_identity, halfRows * halfRows * sizeof(float));
    cudaMemcpy(d_identity, h_identity, halfRows * halfRows * sizeof(float), cudaMemcpyHostToDevice);


    // Solve the linear system A * X = I in one call
    cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, halfRows, halfRows, d_A, halfRows, d_Pivot, d_identity, halfRows, d_Info);

    // Calculate CA^{-1} and store in d_C (assuming d_C initially contains C)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, halfRows, halfRows, halfRows,
                &alpha, d_C, halfRows, d_identity, halfRows, &beta, d_C, halfRows);

    // Calculate (CA^{-1})B and store in d_temp
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, halfRows, halfCols, halfRows,
                &alpha, d_C, halfRows, d_B, halfRows, &beta, d_temp, halfRows);

    // Calculate Q = D - (CA^{-1})B and store the result in d_Q (assuming d_Q initially contains D)
    const float minus_one = -1.0f;
    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, halfRows, halfCols,
                &minus_one, d_temp, halfRows, &alpha, d_D, halfRows, d_Q, halfRows);


    // Compute A^1/2
    cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, halfRows, d_A, halfRows, d_Workspace, Lwork, d_Info);

    // Allocate memory for the matrix L with dimensions rows x rows
    float* d_L;
    cudaMalloc(&d_L, sizeof(float) * cols * rows);
    // Initialize matrix L with zeros
    float* h_zero = (float*)calloc(rows * rows, sizeof(float)); // Allocate and set to zero
    cudaMemcpy(d_L, h_zero, sizeof(float) * rows * rows, cudaMemcpyHostToDevice); // Copy to device

    // Copy A^(1/2) into the top-left block of L
    for (int i = 0; i < halfRows; i++) {
        cudaMemcpy(&d_L[i * rows], &d_A[i * halfRows], sizeof(float) * halfRows, cudaMemcpyDeviceToDevice);
    } // for


    // Perform Cholesky decomposition on A to get L_A
    cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, halfRows, d_A, halfRows, d_Workspace, Lwork, d_Info);

    // Now d_A contains L_A, which is the lower triangular Cholesky factor of A.

    // Solve L_A Y = C for Y (temporarily using d_C for Y)
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                halfRows, halfCols, &alpha, d_A, halfRows, d_C, halfRows);

    // d_C now contains Y, which is L_A^(-1) * C.

    // Copy the result back to d_L into the appropriate position for the lower left block of L
    for (int i = halfRows; i < rows; i++) {
        cudaMemcpy(&d_L[i * rows], &d_C[(i - halfRows) * halfCols], sizeof(float) * halfCols, cudaMemcpyDeviceToDevice);
    } // for


    // Perform Cholesky decomposition on Q to get Q^(1/2)
    // Note: cusolverDnSpotrf will overwrite d_Q with the Cholesky factor, which is Q^(1/2)
    cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, halfCols, d_Q, halfCols, d_Workspace, Lwork, d_Info);

    // d_Q now contains Q^(1/2)
    // Copy Q^(1/2) into the bottom-right block of L
    for (int i = halfRows; i < rows; i++) {
        cudaMemcpy(&d_L[i * rows + halfCols], &d_Q[(i - halfRows) * halfCols], sizeof(float) * halfCols, cudaMemcpyDeviceToDevice);
    } // for

    float *h_L = (float *)malloc(rows * cols * sizeof(float));
    cudaMemcpy(h_L, d_L, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Check if L is lower triangular
    for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            if (h_L[i * cols + j] != 0) {
                printf("L is not lower triangular at (%d, %d)\n", i, j);
                return EXIT_FAILURE;
            }
        }
    }


    // Destroy handles
    cusolverDnDestroy(cusolver_handle);
    cublasDestroy(handle);

    // free host side variables
    free(h_matrix);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    // cudaFree
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    cudaFree(d_Q);
    cudaFree(d_temp);



    return EXIT_SUCCESS;
} // main()
