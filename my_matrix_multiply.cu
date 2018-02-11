#include <stdio.h>
#include <cuda_runtime.h>
#include "my_matmul.h"

__global__ void matmul_kernel(float* lhs, int lhs_rows, int lhs_cols,
	    float* rhs, int rhs_rows, int rhs_cols,
	    float* result) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= lhs_rows * rhs_cols) {
    return;
  }

  
  int result_row = i / rhs_cols;
  int result_col = i % rhs_cols;

  float value = 0;
  for (int j = 0; j != lhs_cols; ++j) {
    float lhs_factor = lhs[result_row * lhs_cols + j];
    float rhs_factor = rhs[j * rhs_cols + result_col];
    value +=  lhs_factor * rhs_factor;
  }
  
  result[i] = value;
}


void matmul_cpu(float* lhs, int lhs_rows, int lhs_cols,
	    float* rhs, int rhs_rows, int rhs_cols,
                     float* result) {
  printf("kernel started\n");
  float* d_lhs = get_cuda_matrix(lhs_rows, lhs_cols, 0);
  float* d_rhs = get_cuda_matrix(rhs_rows, rhs_cols, 0);
  float* d_result = get_cuda_matrix(lhs_rows, rhs_cols, 0);

  cudaMemcpy(d_lhs, lhs, lhs_rows * lhs_cols * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_rhs, rhs, rhs_rows * rhs_cols * sizeof(float),
             cudaMemcpyHostToDevice);

  matmul(d_lhs, lhs_rows, lhs_cols, d_rhs, rhs_rows, rhs_cols, d_result);

  cudaMemcpy(result, d_result, lhs_rows * rhs_cols * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_lhs);
  cudaFree(d_rhs);
  cudaFree(d_result);

  printf("kernel finished\n");
}

void matmul(float* lhs, int lhs_rows, int lhs_cols,
	    float* rhs, int rhs_rows, int rhs_cols,
	    float* result) {
  
  int threadsPerBlock = 256;
  int num_elements = lhs_rows * rhs_cols;
  int blocksPerGrid = (num_elements / threadsPerBlock) + 1;
  matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(lhs, lhs_rows, lhs_cols,
						    rhs, rhs_rows, rhs_cols,
						    result);
  cudaError_t err = cudaSuccess;  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error!\n");
  } else {
    ;//printf("Success!\n");
  }
}  
  

__global__ void initialize_kernel(float* matrix, int num_elements, float value) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= num_elements) {
    return;
  }

  matrix[i] = value;
}



void initialize(float* matrix, int num_elements, float value) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_elements / threadsPerBlock) + 1;
  initialize_kernel<<<blocksPerGrid, threadsPerBlock>>>(matrix, num_elements, value);
  cudaError_t err = cudaSuccess;  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error!\n");
  } else {
    ;//printf("Success!\n");
  }
}

float* get_cuda_matrix(int rows, int cols, float init_value) {
  size_t size = rows * cols * sizeof(float);
  float* d_A = NULL;
  cudaError_t err = cudaSuccess;  
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    printf("error!\n");
  } else {
    ;//printf("Success!\n");
  }

  initialize(d_A, rows * cols, init_value);
  return d_A;
}

void print_matrix(float* matrix, int rows, int cols) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf("%f ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}


void print_cpu(float* matrix, int rows, int cols) {
  size_t size = rows * cols * sizeof(float);
  float* host_matrix = (float*) malloc(size);

  cudaMemcpy(host_matrix, matrix, size, cudaMemcpyDeviceToHost);
  print_matrix(host_matrix, rows, cols);
  free(host_matrix);
}  
  
