#include "helpscreen.h"
#include "ui_helpscreen.h"

HelpScreen::HelpScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::HelpScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Help");
    setGeometry(200, 85, 1500, 900);
}

HelpScreen::~HelpScreen()
{
    delete ui;
}

void HelpScreen::on_btn_closeWindow_clicked()
{
    emit helpScreenClosed();
    this->close();
}
