//
// Program Name:              helpscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the HelpScreen class. See header file for details.
//

#include "startscreen.h"
#include "ui_startscreen.h"
#include "boardscreen.h"

// Constructor
StartScreen::StartScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StartScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");

     // Load logo
    QString path = R"(C:/Users/laesc/OneDrive/Desktop/chester/logos/8bitlogo.png)";
    QPixmap img(path);
    img = img.scaled(img.size()*.9, Qt::KeepAspectRatio);
    ui->lbl_logo->setPixmap(img);

    this->show();
}

// Deconstructor
StartScreen::~StartScreen()
{
    delete ui;
}

// Reopen this form (main menu) whenever a child form emits it's "I'm closing" signal
void StartScreen::on_childScreenClosed()
{
    this->show();
}

// On pressing Play button, open a new instance of the game screen and hide the main menu
void StartScreen::on_btn_play_clicked()
{
    BoardScreen *boardScreen = new BoardScreen(this);
    boardScreen->setWindowState(Qt::WindowMaximized);
    boardScreen->show();

    connect(boardScreen, &BoardScreen::boardScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

// On pressing Help button, open a new instance of the help screen and hide the main menu
void StartScreen::on_btn_help_clicked()
{
    HelpScreen *helpScreen = new HelpScreen(this);
    helpScreen->setWindowState(Qt::WindowMaximized);
    helpScreen->show();

    connect(helpScreen, &HelpScreen::helpScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

// On pressing Settings button, open a new instance of the settings screen and hide the main menu
void StartScreen::on_btn_settings_clicked()
{
    SettingsScreen *settingsScreen = new SettingsScreen(this);
    settingsScreen->setWindowState(Qt::WindowMaximized);
    settingsScreen->show();

    connect(settingsScreen, &SettingsScreen::settingsScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

