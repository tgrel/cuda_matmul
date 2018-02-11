#ifndef MY_MATMUL_H
#define MY_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

void matmul(float* lhs, int lhs_rows, int lhs_cols,
              float* rhs, int rhs_rows, int rhs_cols,
              float* result);

void matmul_cpu(float* lhs, int lhs_rows, int lhs_cols,
              float* rhs, int rhs_rows, int rhs_cols,
              float* result);

void initialize(float* matrix, int num_elements, float value);

float* get_cuda_matrix(int rows, int cols, float init_value);

void print_cpu(float* matrix, int rows, int cols);

#ifdef __cplusplus
}
#endif


#endif
