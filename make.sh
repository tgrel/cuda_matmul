#! /bin/bash -x

nvcc  --shared -Xcompiler  -fPIC -o libmy_matmul.so my_matrix_multiply.cu

g++  -o test test.c  -I '.' -L '.'  -lmy_matmul  -Wl,-rpath,"\$ORIGIN"
