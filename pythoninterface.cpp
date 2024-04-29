#include "pythoninterface.h"

// Python includes
#include "C:/Users/laesc/anaconda3/include/Python.h"
#include <iostream>
#include <filesystem>

#include <QStringConverter>

PythonInterface::PythonInterface()
{
}

QString PythonInterface::getEvaluation(QString UCI)
{
    // Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/python");
    // Py_SetPythonHome(L"C:/Users/laesc/anaconda3");
    Py_Initialize();
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

QString PythonInterface::getNextMove(QString UCI)
{
    // Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/python");
    Py_Initialize();

    QString result;
    std::string str_uci = UCI.toStdString();

    PyObject *pName = PyUnicode_FromString("get_next_move");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    // If the module is found
    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "get_next_best_move");

        // If the function exists and is callable
        if (pFunc && PyCallable_Check(pFunc)) {
            // PyObject* pArgs = PyUnicode_FromString(str_uci.c_str());
            PyObject* pArgs = PyTuple_Pack(1, PyUnicode_FromString(str_uci.c_str()));
            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);

            // If we recieve a return value
            if (pValue != NULL) {
                QString return_value = PyUnicode_AsUTF8(pValue);
                result = return_value;
                Py_DECREF(pValue);
            } else {
                // Else if the function didn't return anything
                PyErr_Print();
                fprintf(stderr, "Python function call failed\n");
            }
        } else {
            // Else if we cannot access the function
            PyErr_Print();
            fprintf(stderr, "Cannot find function get_next_best_move\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        // Else if the module is not found
        PyErr_Print();
        fprintf(stderr, "Failed to load get_next_move\n");
    }

    Py_Finalize();
    return result;
}

void PythonInterface::testPython(QLabel *label)
{
    // Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/python");
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
                QString result = PyUnicode_AsUTF8(pValue);
                // printf("Result of Python function call: %ld\n", result);
                label->setText(result);
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
}

