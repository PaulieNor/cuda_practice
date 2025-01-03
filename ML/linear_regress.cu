#include <iostream>
#include <cuda_runtime.h>


typedef struct {
    float m;
    float c;
} linear_regress;

typedef struct {
    float sum_x;
    float sum_y;
    float sum_x_squared;
    float sum_xy;
} sums;



__global__ void linearRegressionKernel(float *x, float *y, sums *shared_sums, int n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    if (idx < n) {
        atomicAdd(&(shared_sums->sum_x), x[idx]);
        atomicAdd(&(shared_sums->sum_y), y[idx]);
        atomicAdd(&(shared_sums->sum_x_squared), x[idx] * x[idx]);
        atomicAdd(&(shared_sums->sum_xy), x[idx] * y[idx]);
    }

}

__host__ int initialize_arrays(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }

    return 0;
}

__host__ linear_regress calculate_regression(sums sums, int n) {
    linear_regress result;

    result.m = ((n * sums.sum_xy) - (sums.sum_x * sums.sum_y)) / ((n * sums.sum_x_squared) - (sums.sum_x * sums.sum_x));
    result.c = (sums.sum_y - (result.m * sums.sum_x)) / n;

    return result;

}

int main() {

    // Initialise array.

    int n = 1024;

    float x[n];
    float y[n];

    initialize_arrays(x, y, n);

    float *d_x, *d_y;

    sums shared_sums = {0.0f, 0.0f, 0.0f, 0.0f};
    sums *d_shared_sums;

    size_t size = n * sizeof(float); // Size of each array in bytes


    // Number of threads in each thread block
    int threadsPerBlock = 256;

    // Number of thread blocks in grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;


    // Allocate device memory
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&d_shared_sums, sizeof(shared_sums));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shared_sums, &shared_sums, sizeof(shared_sums), cudaMemcpyHostToDevice);




    // Launch kernel on the GPU
    linearRegressionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_shared_sums, n);

    cudaMemcpy(&shared_sums, d_shared_sums, sizeof(shared_sums), cudaMemcpyDeviceToHost);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        return EXIT_FAILURE;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Failed to synchronize (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        return EXIT_FAILURE;
    }

    // Free device memory

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_shared_sums);

    // Reset the device and exit
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "Failed to deinitialize the device! error=" << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    linear_regress linear_regress = calculate_regression(shared_sums, n);


    std::cout << "Results:\n";

    std::cout << "m: " << linear_regress.m << std::endl;
    std::cout << "c: " << linear_regress.c << std::endl;



    return 0;
}