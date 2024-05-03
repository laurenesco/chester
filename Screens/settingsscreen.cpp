//
// Program Name:              settingsscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the SettingsScreen class. See header file for details.
//

#include "settingsscreen.h"
#include "ui_settingsscreen.h"

// Constructor
SettingsScreen::SettingsScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SettingsScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Settings");
    this->config = new Config();
    config->refreshConfig();
    qDebug() << "Opening settings screen: Assist mode:" << config->getAssistModeOn() << "Color:" << config->getColor() << "Difficulty:" << config->getDifficulty();

    fillComboBoxes();
}

// Deconstructor
SettingsScreen::~SettingsScreen()
{
    delete ui;
}

void SettingsScreen::fillComboBoxes()
{
    // Difficulty combobox
    QStringList list = {"Easy", "Medium", "Hard"};
    ui->cmb_difficulty->addItems(list);
    int index = config->getDifficulty() - 1;
    qDebug() << "Difficulty index:" << index;
    ui->cmb_difficulty->setCurrentIndex(index);

    // Player color
    list = {"White", "Black"};
    ui->cmb_color->addItems(list);
    index = config->getColor() == true ? 0 : 1;
    qDebug() << "Color index:" << index;
    ui->cmb_color->setCurrentIndex(index);

    // Assisted mode
    list = {"On", "Off"};
    ui->cmb_assist->addItems(list);
    index = config->getAssistModeOn() == true ? 0 : 1;
    qDebug() << "Assist mode state:" << config->getAssistModeOn() << "Assisted mode:" << index;
    ui->cmb_assist->setCurrentIndex(index);
}

// On pressing Close button, emit closing signal and close this form
void SettingsScreen::on_btn_closeWindow_clicked()
{
    qDebug() << "Closing settings screen: Assist mode:" << config->getAssistModeOn() << "Color:" << config->getColor() << "Difficulty:" << config->getDifficulty();
    Q_EMIT settingsScreenClosed();
    this->close();
}

// On difficulty change
void SettingsScreen::on_cmb_difficulty_currentTextChanged(const QString &arg1)
{

}

// On color changed
void SettingsScreen::on_cmb_color_currentTextChanged(const QString &arg1)
{

}

// On assist mode changed
void SettingsScreen::on_cmb_assist_currentTextChanged(const QString &arg1)
{

}


void SettingsScreen::on_cmb_assist_activated(int index)
{
    // Possible options list = {"On", "Off"};
    bool active = index == 0 ? true : false;
    config->setAssistModeOn(active);
    config->saveConfig();
    return;
}


void SettingsScreen::on_cmb_color_activated(int index)
{
    // Possible options {"White", "Black"};
    bool isWhite = index == 0 ? true : false;
    config->setColor(isWhite);
    config->saveConfig();
    return;
}


void SettingsScreen::on_cmb_difficulty_activated(int index)
{
    // Possible options {"Easy", "Medium", "Hard"};
    if (index == 0) {
        config->setDifficulty(1);
    } else if (index == 2) {
        config->setDifficulty(2);
    } else {
        config->setDifficulty(3);
    }

    config->saveConfig();
    return;
}

