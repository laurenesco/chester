//
// Program Name:              startscreen.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the StartScreen class. This class is responsible for:
//                                              - Generating the main landing page, or "start screen" of the application
//

/*
- signals and slots
- global config file
- json holding config params
- ml update
- headers
*/

#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QMainWindow>
#include <QLabel>
#include <QMovie>
#include <QObject>
#include "boardscreen.h"
#include "helpscreen.h"
#include "settingsscreen.h"

namespace Ui {
class StartScreen;
}

class StartScreen : public QMainWindow
{
    Q_OBJECT

public:
    explicit StartScreen(QWidget *parent = nullptr);
    ~StartScreen();

private Q_SLOTS:
    void on_btn_play_clicked(); // Open the board screen
    void on_childScreenClosed(); // Reopen main window
    void on_btn_help_clicked(); // Open help screen
    void on_btn_settings_clicked(); // Open settings screen

private:
    Ui::StartScreen *ui;
};

#endif // STARTSCREEN_H
