//#include <Python.h>
#include <C:/Users/laesc/AppData/Local/Programs/Python/Python310/include/Python.h>
#include <iostream>
#include <filesystem>

int main() {

    Py_SetProgramName(L"/home/lauren/chester/ml");
    Py_Initialize();

    PyObject *pName = PyUnicode_FromString("my_script");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "add_numbers");

        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject *pArgs = PyTuple_Pack(2, PyLong_FromLong(3), PyLong_FromLong(4));
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != NULL) {
                long result = PyLong_AsLong(pValue);
                printf("Result of Python function call: %ld\n", result);
                Py_DECREF(pValue);
            } else {
                PyErr_Print();
                fprintf(stderr, "Call failed\n");
            }
        } else {
            PyErr_Print();
            fprintf(stderr, "Cannot find function \"add_numbers\"\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"my_script\"\n");
    }

    Py_Finalize();
    return 0;
}

// g++ main.cpp -I /usr/include/python3.10 -L /usr/lib/python3.10 -lpython3.10 -o call_python

// ./call_python

// windows
// g++ testing.cpp -I C:/Users/laesc/AppData/Local/Programs/Python/Python310/include -L C:/Users/laesc/AppData/Local/Programs/Python/Python310/libs -lpython310 -o windows_call_python

// ./call_python
