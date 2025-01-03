#include <cuda_runtime.h>
#include <iostream>

// Kernel function to add two arrays
__global__ void add_arrays(const float *A, const float *B, float *C, int n) {
    // Implement the addition here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
        printf("Thread %d: A[%d] = %f, B[%d] = %f\n", idx, idx, A[idx], idx, B[idx]);
    }
}

// Utility function to initialize arrays
void initialize_arrays(float *A, float *B, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }
}

// Host function to run the kernel
void add_arrays_cuda(const float *A, const float *B, float *C, int n) {
    // Allocate device memory

    float *d_A, *d_B, *d_C;
    size_t size = n * sizeof(float); // Size of each array in bytes

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);


    // Copy input arrays to the device

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);


    // Launch kernel with appropriate block and grid size

    int threads_per_block = 256;
    int blocks = (n + threads_per_block + 1)/ threads_per_block;

    add_arrays<<<blocks, threads_per_block>>>(d_A, d_B, d_C, n);

    // Copy the result array back to the host

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);


    // Free device memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main() {
    const int N = 1024; // Array size

    // Allocate host memory
    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    // Initialize input arrays
    initialize_arrays(A, B, N);

    // Perform addition on the GPU
    add_arrays_cuda(A, B, C, N);

    // Display results (optional)
    std::cout << "Results:\n";
    for (int i = 0; i < 10; i++) { // Display first 10 results
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
