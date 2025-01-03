#include <cuda_runtime.h>
#include <iostream>

__global__ void create_histogram(const int* data, int* histogram, int data_size, int n_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < data_size; i += stride) {
        int bin = data[i] % n_bins;
        atomicAdd(&histogram[bin], 1);
    }
}


void cuda_histogram(const int* h_data, int* h_histogram, int data_size, int n_bins) {

    int* d_data;
    int* d_histogram;

    cudaMalloc(&d_data, data_size * sizeof(int));
    cudaMalloc(&d_histogram, n_bins * sizeof(int));

    cudaMemcpy(d_data, h_data, data_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_histogram, h_histogram, n_bins * sizeof(int), cudaMemcpyHostToDevice);


    int threads_per_block = 256;
    int blocks_per_grid = (data_size - 1 + threads_per_block) / threads_per_block;

    create_histogram<<<blocks_per_grid, threads_per_block>>>(d_data, d_histogram, data_size, n_bins);

    cudaMemcpy(h_histogram, d_histogram, n_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histogram);

}

int main(){
    
    const int data_size = 1000;
    const int n_bins = 10;

    int h_data[data_size];

    for (int i = 0; i < data_size; i++) {
        h_data[i] = i % data_size;

    }

    int h_histogram[n_bins] = {0};

    cuda_histogram(h_data, h_histogram, data_size, n_bins);


    for (int i = 0; i < n_bins; i++) {
        std::cout << "Bin " << i << ": " << h_histogram[i] << std::endl;
    }

    return 0;

}