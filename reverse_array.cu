#include <cuda_runtime.h>
#include <iostream>

// Kernel function to add two arrays
__global__ void reverse_array(float *A, int n) {
    // Implement the addition here
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n/2) {
        int temp = A[idx];
        A[idx] = A[n-1-idx];
        A[n-1-idx] = temp;
    }
}

// Utility function to initialize arrays
void initialize_arrays(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
    }
}

// Host function to run the kernel
void reverse_array_cuda(float *A, int n) {
    // Allocate device memory

    float *d_A;
    size_t size = n * sizeof(float); // Size of each array in bytes

    cudaMalloc((void **)&d_A, size);


    // Copy input arrays to the device

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);


    // Launch kernel with appropriate block and grid size

    int threads_per_block = 256;
    int blocks = (n + threads_per_block + 1)/ threads_per_block;

    reverse_array<<<blocks, threads_per_block>>>(d_A, n);

    // Copy the result array back to the host

    cudaMemcpy(A, d_A, size, cudaMemcpyDeviceToHost);


    // Free device memory

    cudaFree(d_A);

}

int main() {
    const int N = 1024; // Array size

    // Allocate host memory
    float *A = new float[N];

    // Initialize input arrays
    initialize_arrays(A, N);

    // Perform addition on the GPU
    reverse_array_cuda(A, N);

    // Display results (optional)
    std::cout << "Results:\n";
    for (int i = 0; i < N; i++) { // Display first 10 results
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    delete[] A;
    return 0;
}
