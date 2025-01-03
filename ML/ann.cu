#include <cuda_runtime.h>
#include <iostream>
#include <math.h>



__global__ void matmul(const float *A, const float *B, float *C, int n) {

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (column < n && row < n) {
        float value = 0.0f;
        for (int i = 0; i < n; i++) {
            value = A[row * n + i] * B[i * n + column];
        }
        C[row * n + column] = value;
    }
}

__global__ void relu(float *matrix, const int n) {
    // ReLU activation function.

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        matrix[idx] = fmaxf(0.0f, matrix[idx]);
    }
}

__host__ float mse(float *y_true, float *y_pred, int n, int threadsPerBlock, int blocksPerGrid) {
    float mse = 0.0f;
    mse_total<<<blocksPerGrid,threadsPerBlock>>>(y_true, y_pred, n, mse, 0);
    return mse/n;
}

__global__ void mse_total(float *y_true, float *y_pred, int n, float mse_total, int idx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < n) {
        atomicAdd(&mse_total, (y_true[idx] - y_pred[idx]) * (y_true[idx] - y_pred[idx]));
    }
    
}

__global__ void mse_update_gradients(float *y_true, float *y_pred, float *gradients, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        gradients[idx] = 2 * (y_pred[idx] - y_true[idx]) / n;
    }

}

__global__ void update_weights_and_biases(
    float *weights, float *biases, const float *weight_gradients, const float *bias_gradients, float learning_rate, int n_inputs, int n_outputs) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int column = threadIdx.y + blockIdx.y * blockDim.y;


    if (row < n_outputs && column < n_inputs) {
        weights[row * n_inputs + column] -= learning_rate * weight_gradients[row * n_inputs + column];
    }

    if (row < n_outputs) {
        biases[row] -= learning_rate * bias_gradients[row];
    }
}

int main() {
    // Set up device
    int device = 0;
    cudaSetDevice(device);

    const int input_size = 5;
    const int hidden_layer_size = 4;
    const int output_size = 3;

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Synchronize
    cudaDeviceSynchronize();

    return 0;
}