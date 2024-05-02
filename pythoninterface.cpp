#include "pythoninterface.h"

// Python includes
// #include "C:/Users/laesc/anaconda3/include/Python.h"
#include "C:/Users/laesc/AppData/Local/Programs/Python/Python312/include/Python.h"
#include <iostream>
#include <filesystem>
#include <Windows.h>

#include <QStringConverter>

PythonInterface::PythonInterface()
{
    putenv("PYTHONWIN_HIDE_CONSOLE=1");
    SetEnvironmentVariableW(L"PYTHONWIN_HIDE_CONSOLE", L"1");
}

QString PythonInterface::getStats(QString UCI) {
    Py_Initialize();
    putenv("PYTHONWIN_HIDE_CONSOLE=1");
    SetEnvironmentVariableW(L"PYTHONWIN_HIDE_CONSOLE", L"1");

    QString result;
    std::string str_uci = UCI.toStdString();

    PyObject *pName = PyUnicode_FromString("get_stats");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    // If the module is found
    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "get_game_stats");

        // If the function exists and is callable
        if (pFunc && PyCallable_Check(pFunc)) {
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
            fprintf(stderr, "Cannot find function get_board_evaluation\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        // Else if the module is not found
        PyErr_Print();
        fprintf(stderr, "Failed to load get_evaluation\n");
    }

    Py_Finalize();
    return result;
}

QString PythonInterface::getEvaluation(QString UCI)
{
    Py_Initialize();
    putenv("PYTHONWIN_HIDE_CONSOLE=1");
    SetEnvironmentVariableW(L"PYTHONWIN_HIDE_CONSOLE", L"1");

    QString result;
    std::string str_uci = UCI.toStdString();

    PyObject *pName = PyUnicode_FromString("get_evaluation");
    PyObject *pModule = PyImport_Import(pName);
    Py_DECREF(pName);
    // If the module is found
    if (pModule != NULL) {
        PyObject *pFunc = PyObject_GetAttrString(pModule, "get_board_evaluation");

        // If the function exists and is callable
        if (pFunc && PyCallable_Check(pFunc)) {
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
            fprintf(stderr, "Cannot find function get_board_evaluation\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } else {
        // Else if the module is not found
        PyErr_Print();
        fprintf(stderr, "Failed to load get_evaluation\n");
    }

    Py_Finalize();
    return result;
}

QString PythonInterface::getNextMove(QString UCI)
{
    Py_Initialize();
    putenv("PYTHONWIN_HIDE_CONSOLE=1");
    SetEnvironmentVariableW(L"PYTHONWIN_HIDE_CONSOLE", L"1");

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
