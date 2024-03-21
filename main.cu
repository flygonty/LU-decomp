#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;


void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

void readMatrixFromFile(const std::string& filename, std::vector<float>& matrix, int& n) {
    std::ifstream file(filename.c_str());
    std::string line;
    n = 0; // Initialize n to 0, will count the number of rows

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            matrix.push_back(value);
            // std::cout << value << "\n";
            n++;
        } // while
        
    } // while
} // readMatrixFile()

int main() {
    Timer timer;
    std::vector<float> host_matrix;
    int n; // Dimension of the square matrix

    readMatrixFromFile("../matrix-multiply-flygonty/data", host_matrix, n);


    int totalElements = n;
    if (host_matrix.size() != totalElements) {
        std::cerr << "The matrix is not square or the file is incorrectly formatted." << std::endl;
        return 1;
    } // if

    // Allocate memory on the device for the matrix
    float* d_matrix = NULL;
    cudaMalloc(&d_matrix, totalElements * sizeof(float));

    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, host_matrix.data(), totalElements * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize cuSolver
    cusolverDnHandle_t cusolver_handle = NULL;
    cusolverDnCreate(&cusolver_handle);

    // Allocate workspace for LU decomposition
    int workspace_size = 0;
    cusolverDnSgetrf_bufferSize(cusolver_handle, n, n, d_matrix, n, &workspace_size);

    float* workspace;
    cudaMalloc(&workspace, workspace_size * sizeof(float));

    // Create device array for pivot indices and LU decomposition info
    int *devIpiv, *devInfo;
    cudaMalloc(&devIpiv, n * sizeof(int));
    cudaMalloc(&devInfo, sizeof(int));

    // Perform LU decomposition
    startTime(&timer);
    cusolverDnSgetrf(cusolver_handle, n, n, d_matrix, n, workspace, devIpiv, devInfo);
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Cleanup
    cudaFree(workspace);
    cudaFree(devIpiv);
    cudaFree(devInfo);
    cudaFree(d_matrix);
    cusolverDnDestroy(cusolver_handle);

    return 0;

} // main()
