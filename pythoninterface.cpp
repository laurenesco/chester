#include "pythoninterface.h"

// Python includes
#include "C:/Users/laesc/AppData/Local/Programs/Python/Python310/include/Python.h"
#include <iostream>
#include <filesystem>

#include <QStringConverter>

PythonInterface::PythonInterface()
{
    Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/python");
    Py_Initialize();
}

QString PythonInterface::getEvaluation(QString UCI)
{
    PyObject *pName = PyUnicode_FromString("get_evaluation");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "get_evaluation");

        if (pFunc && PyCallable_Check(pFunc)) {
            std::string uci = UCI.toStdString();
            const char* cstr = uci.c_str();
            PyObject* pArgs = PyUnicode_DecodeUTF8(cstr, uci.size(), "strict");
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            if (pValue != NULL) {
                QString result = PyUnicode_AsUTF8(pValue);
                Py_DECREF(pValue);
                return result;
            } else {
                PyErr_Print();
                fprintf(stderr, "Call failed");
                return "Error: Function call failed";
            }
        } else {
            PyErr_Print();
            fprintf(stderr, "Cannot find function.");
            return "Error: Cannot find function";
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load module.");
        return "Error: Failed to load module";
    }
    Py_Finalize();
    return "Error";
}
