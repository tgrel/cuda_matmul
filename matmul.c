/*
 * author: tomasz.grel@gmail.com
 * date 2017-04-08
 */

#include <Python.h>
#include <stdio.h>
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
 
static PyMethodDef Methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


/*
 * initialization function installing the handler
 */ 
PyMODINIT_FUNC
initcudamatmul(void) {
  (void) Py_InitModule("cbacktrace", Methods);
  printf("module initialized from C");
}

