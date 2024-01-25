#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QMainWindow>
#include <QLabel>
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

private slots:
    void on_btn_play_clicked(); // Open the board screen
    void on_childScreenClosed(); // Reopen main window
    void on_btn_help_clicked(); // Open help screen
    void on_btn_settings_clicked(); // Open settings screen

private:
    Ui::StartScreen *ui;
};

#endif // STARTSCREEN_H
