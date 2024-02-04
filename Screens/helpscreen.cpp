//
// Program Name:              helpscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the HelpScreen class. See header file for details.
//

#include "helpscreen.h"
#include "ui_helpscreen.h"


// Python includes
#include "C:/Users/laesc/AppData/Local/Programs/Python/Python310/include/Python.h"
#include <iostream>
#include <filesystem>

HelpScreen::HelpScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::HelpScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Help");
    setGeometry(200, 85, 1500, 900);


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
                QString result = PyUnicode_AsUTF8(pValue);
                // printf("Result of Python function call: %ld\n", result);
                ui->lbl_testPython->setText(result);
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

HelpScreen::~HelpScreen()
{
    delete ui;
}

void HelpScreen::on_btn_closeWindow_clicked()
{
    Q_EMIT helpScreenClosed();
    this->close();
}
