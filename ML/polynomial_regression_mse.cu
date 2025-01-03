#include <iostream>
#include <cuda_runtime.h>
#include <vector>

__device__ void predict(float x, float* y_pred, float* coefficients, int degree) {
    //printf("Predict Thread %d: x = %f\n", threadIdx.x);
    *y_pred = 0.0f;
    for (int idx = 0; idx < degree; ++idx) {
        float term = coefficients[idx] * powf(x, idx);
        if (term > 1e20f || term < -1e20f) {
            term = 0.0f;  // Prevent overflow
        }
        *y_pred += term;
    }
}

__global__ void calculate_gradient_kernel(float* x, float* y, float* gradients, float* coefficients, int degree, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("Calculate Gradient Thread %d: x[%d] = %f, y[%d] = %f\n", idx, idx, x[idx], idx, y[idx]);
    if (idx < degree) {
        gradients[idx] = 0.0f;
        for (int i = 0; i < n; i++) {
            float y_pred = 0.0f;
            predict(x[i], &y_pred, coefficients, degree);
            gradients[idx] += (y_pred - y[i]) * powf(x[i], idx);
        }
    }
}

__global__ void update_coefficients_kernel(float* coefficients, float* gradients, float training_rate, int degree) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    //printf("Update Coefficients Thread %d: coefficients[%d] = %f, gradients[%d] = %f\n", idx, idx, coefficients[idx], idx, gradients[idx]);
    if (idx < degree) {
        coefficients[idx] -= training_rate * gradients[idx];
    }
}

__global__ void calculate_mse_kernel(float* x, float* y, float* mse, float* coefficients, int degree, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("MSE Thread %d: x[%d] = %f, y[%d] = %f\n", idx, idx, x[idx], idx, y[idx]);
    if (idx < n) {
        float y_pred = 0.0f;
        predict(x[idx], &y_pred, coefficients, degree);
        printf("y_pred = %f, y = %f, mse = %f \n", y_pred, y[idx], (y_pred - y[idx]) * (y_pred - y[idx]));
        atomicAdd(mse, (y_pred - y[idx]) * (y_pred - y[idx]));
    }
}

int main() {
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device Name: " << prop.name << std::endl;

    // Allocate and initialize host data
    const int N = 50;
    const int degree = 3;
    const float training_rate = 0.00000001;
    float h_x[N] = {0};            // Input array
    float h_y[N] = {0};            // Output array (targets)
    float coefficients[degree] = {0};   // Polynomial coefficients (weights)
    float gradients[32] = {0};           // Gradients array (for updates)    
    float h_mse = 0.0f;
    float* d_mse;
    cudaMalloc((void**)&d_mse, sizeof(float));

    // Initialize x and y with random values
    for (int i = 0; i < N; ++i) {
        h_x[i] = static_cast<float>(i);
    }

    // Generate y values based on a polynomial curve with some added noise
    for (int i = 0; i < N; ++i) {
        h_y[i] = powf(h_x[i], 2) + powf(h_x[i], 3) + h_x[i];  // Polynomial with noise
        printf("[%f, %f]", h_x[i],  h_y[i]);
    }

    printf("\n");

    // Initialize coefficients
    for (int i = 0; i < degree; ++i) {
        coefficients[i] = 1.0f;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_coefficients, *d_gradients;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_coefficients, degree * sizeof(float));
    cudaMalloc((void**)&d_gradients, 32 * sizeof(float));

    // Transfer data from host to device
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Number of threads in each thread block
    int threadsPerBlock = 256;

    // Number of thread blocks in grid
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Kernel launch with data
    
    for (int epoch = 1; epoch <= 10000; ++epoch) {
        std::cout << "Epoch: " << epoch << std::endl;

        // Copy the current coefficients to device
        cudaMemcpy(d_coefficients, coefficients, degree * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gradients, gradients, 32 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mse, &h_mse, sizeof(float), cudaMemcpyHostToDevice);

        // Launch gradient calculation kernel
        calculate_gradient_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_gradients, d_coefficients, degree, N);
        cudaDeviceSynchronize();

        // Launch coefficient update kernel
        update_coefficients_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_coefficients, d_gradients, training_rate, degree);
        cudaDeviceSynchronize();

        // Launch MSE calculation kernel
        calculate_mse_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_mse, d_coefficients, degree, N);
        cudaDeviceSynchronize();

        // Copy MSE and coefficients back to host
        cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(coefficients, d_coefficients, degree * sizeof(float), cudaMemcpyDeviceToHost);

        // Print MSE and coefficients for each epoch
        std::cout << "MSE: " << h_mse/N << std::endl;
        for (int i = 0; i < degree; ++i) {
            std::cout << "Coefficient " << i << ": " << coefficients[i] << std::endl;
        }

        h_mse = 0.0f;
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_coefficients);
    cudaFree(d_mse);
    cudaFree(d_gradients);



    return 0;
}
