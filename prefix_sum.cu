#include <cuda_runtime.h>
#include <iostream>

// Kernel function to prefix sum.
__global__ void prefix_sum_array(float *array, float output, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;

    // Copy input to shared memory.

    temp[tid] = array[tid];

    // Upsweep phase - building the tree.

    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < n) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Clear the last element.

    if (tid == 0) {
        temp[n-1] = 0;
    }
    __syncthreads();

    // Downsweep phase - traversing the tree.

    for (int stride = n/2; stride > 0; stride /= 2) {
        int index = (tid + 1) * 2 * stride - 1;
        if (index < n) {
            float temp_val = temp[index];
            temp[index] += temp[index - stride];
            temp[index - stride] = temp_val;
        }
        __syncthreads();
    }

}

// Utility function to initialize arrays
void initialize_arrays(float *A, int n) {
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
    }
}

// Host function to run the kernel
void prefix_sum_array_cuda(float *A, int n) {
    // Allocate device memory

    float *d_A;
    size_t size = n * sizeof(float); // Size of each array in bytes

    cudaMalloc((void **)&d_A, size);


    // Copy input arrays to the device

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);


    // Launch kernel with appropriate block and grid size

    int threads_per_block = 256;
    int blocks = (n + threads_per_block + 1)/ threads_per_block;

    prefix_sum_array<<<blocks, threads_per_block>>>(d_A, n);

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
    prefix_sum_array_cuda(A, N);

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
