/*
 * author: tomasz.grel@gmail.com
 * date 2018-02-11
 */

#include <Python.h>
#include <stdio.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include "my_matmul.h"


static PyObject* py_matmul (PyObject* self, PyObject* args) {
  long long lhs, rhs, result;
  int lhs_rows, lhs_cols, rhs_rows, rhs_cols;

  if (!PyArg_ParseTuple(args, "LiiLiiL", &lhs, &lhs_rows, &lhs_cols,
                        &rhs, &rhs_rows, &rhs_cols,
                        &result)) {
        return NULL;
  }

  float* lhs_data = (float*) lhs;
  float* rhs_data = (float*) rhs;
  float* result_data = (float*) result;

  matmul_cpu(lhs_data, lhs_rows, lhs_cols,
             rhs_data, rhs_rows, rhs_cols,
             result_data);
  
  Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"matmul",  py_matmul, METH_VARARGS, "CUDA matrix multiplication"},  
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


/*
 * initialization function installing the handler
 */ 
PyMODINIT_FUNC
initcuda_matmul(void) {
  (void) Py_InitModule("cuda_matmul", Methods);
  printf("module initialized from C\n");
}

