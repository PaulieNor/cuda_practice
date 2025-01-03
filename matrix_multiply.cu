#include <iostream>
#include <cstdlib>
#include <ctime>

#define N 64  // Size of the matrix, you can change it for different sizes

// Kernel function to multiply two matrices
__global__ void matrixMulKernel(float *A, float *B, float *C, int width) {
    // TODO: Calculate the row and column index for each thread

    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < width && column < width) {
        float value = 0.0f;
        for (int i = 0; i < width; ++i) {
            value += A[row * width + i] * B[i * width + column];
        }
        C[row * width + column] = value;
    }

    // TODO: Implement matrix multiplication logic to compute one element of the result matrix C
}

// Host function to initialize matrices with random values
void initializeMatrix(float *matrix, int width) {
    for (int i = 0; i < width * width; ++i) {
        matrix[i] = rand() % 10;  // Random values between 0 and 9
    }
}

// Host function to print the matrix (optional for small sizes)
void printMatrix(float *matrix, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int width = N;

    // Allocate host memory for matrices
    float *h_A = (float *)malloc(width * width * sizeof(float));
    float *h_B = (float *)malloc(width * width * sizeof(float));
    float *h_C = (float *)malloc(width * width * sizeof(float));

    // Initialize matrices with random values
    initializeMatrix(h_A, width);
    initializeMatrix(h_B, width);

    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, width * width * sizeof(float));
    cudaMalloc((void **)&d_B, width * width * sizeof(float));
    cudaMalloc((void **)&d_C, width * width * sizeof(float));

    cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // TODO: Define block size and grid size
    dim3 blockSize(16, 16);  // Example block size, you can adjust this
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (width + blockSize.y - 1) / blockSize.y);

    // Launch the matrix multiplication kernel
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);

    // Copy the result back to host (not needed as we are using host memory)
 
    cudaMemcpy(h_C, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result (optional for small matrices)

    printf("Array A \n");    
    printMatrix(h_A, width);

    printf("Array B \n");
    printMatrix(h_B, width);

    printf("Result \n");
    printMatrix(h_C, width);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
