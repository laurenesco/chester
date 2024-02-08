//
// Program Name:              helpscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the HelpScreen class. See header file for details.
//

#include "helpscreen.h"
#include "ui_helpscreen.h"
#include "engineinterface.h"

HelpScreen::HelpScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::HelpScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Help");
    setGeometry(200, 85, 1500, 900);

    // Test python connection
    EngineInterface engine;
    QString output = engine.runTest();
    ui->lbl_testPython->setText(output);

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
