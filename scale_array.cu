#include <cuda_runtime.h>
#include <iostream>

// Kernel function to add two arrays
__global__ void scale_arrays(const float *A, float *B, int n) {
    // Implement the addition here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        B[idx] = A[idx] + 100;
    }
}

// Utility function to initialize arrays
void initialize_arrays(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
    }
}

// Host function to run the kernel
void scale_arrays_cuda(const float *A, float *B, int n) {
    // Allocate device memory

    float *d_A, *d_B;
    size_t size = n * sizeof(float); // Size of each array in bytes

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);


    // Copy input arrays to the device

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);


    // Launch kernel with appropriate block and grid size

    int threads_per_block = 256;
    int blocks = (n + threads_per_block + 1)/ threads_per_block;

    scale_arrays<<<blocks, threads_per_block>>>(d_A, d_B, n);

    // Copy the result array back to the host

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);


    // Free device memory

    cudaFree(d_A);
    cudaFree(d_B);

}

int main() {
    const int N = 1024; // Array size

    // Allocate host memory
    float *A = new float[N];
    float *B = new float[N];

    // Initialize input arrays
    initialize_arrays(A, N);

    // Perform addition on the GPU
    scale_arrays_cuda(A, B, N);

    // Display results (optional)
    std::cout << "Results:\n";
    for (int i = 0; i < 10; i++) { // Display first 10 results
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] A;
    delete[] B;

    return 0;
}
