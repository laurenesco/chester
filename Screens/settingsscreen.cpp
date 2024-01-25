#include "settingsscreen.h"
#include "ui_settingsscreen.h"

SettingsScreen::SettingsScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SettingsScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Settings");
    setGeometry(200, 85, 1500, 900);
}

SettingsScreen::~SettingsScreen()
{
    delete ui;
}

void SettingsScreen::on_btn_closeWindow_clicked()
{
    emit settingsScreenClosed();
    this->close();
}
