#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "numpy/npy_3kcompat.h"

/*
 * Jomega.c
 * Implements the following function
 * F(x,y) = x/x+y
 * as a ufunc for easier broadcasting in numpy.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 *
 */


static PyMethodDef JomegaMethods[] = {
        {NULL,
        NULL,
        0,
        NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

static void long_double_Jomega(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    long double tmp1, tmp2;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp1 = *(long double *)in1;
        tmp2 = *(long double *)in2;
        *((long double *)out) = tmp1/(tmp1*tmp1+tmp2*tmp2);
        /*END main ufunc computation*/
        in1 += in1_step; in2 += in2_step; out += out_step;
    }
}

static void double_Jomega(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    double tmp1, tmp2;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp1 = *(double *)in1;
        tmp2 = *(double *)in2;
        *((double *)out) = tmp1/(tmp1*tmp1+tmp2*tmp2);
        /*END main ufunc computation*/
        in1 += in1_step; in2 += in2_step; out += out_step;
    }
}

static void float_Jomega(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    float tmp1, tmp2;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp1 = *(float *)in1;
        tmp2 = *(float *)in2;
        *((float *)out) = tmp1/(tmp1*tmp1+tmp2*tmp2);
        /*END main ufunc computation*/
        in1 += in1_step; in2 += in2_step; out += out_step;
    }
}

static void half_float_Jomega(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1], *out = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1], out_step = steps[2];
    float tmp1, tmp2;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        tmp1 = *(npy_half *)in1; tmp1 = npy_half_to_float(tmp1);
        tmp2 = *(npy_half *)in2; tmp2 = npy_half_to_float(tmp2);
        *((float *)out) = npy_float_to_half( tmp1/(tmp1*tmp1+tmp2*tmp2) );
        /*END main ufunc computation*/
        in1 += in1_step; in2 += in2_step; out += out_step;
    }
}

/*This a pointer to the above functions*/
PyUFuncGenericFunction funcs[4] = {&half_float_Jomega,
                                   &float_Jomega,
                                   &double_Jomega,
                                   &long_double_Jomega};

/* These are the input and return dtypes of Jomega.*/

static char types[12] = {
    NPY_HALF,  NPY_HALF,  NPY_HALF,
    NPY_FLOAT, NPY_FLOAT, NPY_FLOAT,
    NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
    NPY_LONGDOUBLE, NPY_LONGDOUBLE, NPY_LONGDOUBLE};

static void *data[4] = {NULL,NULL,NULL,NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    JomegaMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *Jomega, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    Jomega = PyUFunc_FromFuncAndData(funcs, data, types, 4, 2, 1,
                                    PyUFunc_None, "Jomega",
                                    "Jomega_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "Jomega", Jomega);
    Py_DECREF(Jomega);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *Jomega, *d;


    m = Py_InitModule("npufunc", JomegaMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    Jomega = PyUFunc_FromFuncAndData(funcs, data, types, 4, 2, 1,
                                    PyUFunc_None, "Jomega",
                                    "Jomega_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "Jomega", Jomega);
    Py_DECREF(Jomega);
}
#endif
