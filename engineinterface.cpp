#include "engineinterface.h"

// Constructor
EngineInterface::EngineInterface()
{

}

// Function that runs connectivity test using Python.h
QString EngineInterface::runTest() {
    QString result;

    Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/ml");
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
                result = PyUnicode_AsUTF8(pValue);
                // printf("Result of Python function call: %ld\n", result);
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
    return result;
}
