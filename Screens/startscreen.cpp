#include "startscreen.h"
#include "ui_startscreen.h"
#include "boardscreen.h"

StartScreen::StartScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StartScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);

     // Load logo
    QString path = R"(C:/Users/laesc/OneDrive/Desktop/chester/logos/chesster.png)";
    QPixmap img(path);
    ui->lbl_logo->setPixmap(img);

    this->show();
}

StartScreen::~StartScreen()
{
    delete ui;
}

void StartScreen::on_childScreenClosed()
{
    this->show();
}

// Play button clicked
void StartScreen::on_btn_play_clicked()
{
    BoardScreen *boardScreen = new BoardScreen(this);
    boardScreen->show();

    connect(boardScreen, &BoardScreen::boardScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

// Help button clicked
void StartScreen::on_btn_help_clicked()
{
    HelpScreen *helpScreen = new HelpScreen(this);
    helpScreen->show();

    connect(helpScreen, &HelpScreen::helpScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

// Settings button clicked
void StartScreen::on_btn_settings_clicked()
{
    SettingsScreen *settingsScreen = new SettingsScreen(this);
    settingsScreen->show();

    connect(settingsScreen, &SettingsScreen::settingsScreenClosed, this, &StartScreen::on_childScreenClosed);

    this->hide();
}

