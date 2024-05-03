//
// Program Name:              settingsscreen.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the SettingsScreen class. This class is responsible for:
//                                              - Generating the settings page of the application
//                                              - Managing the following settings:
//                                                      - Color scheme
//                                                      - Difficulty
//                                                      - Player Color
//

#ifndef SETTINGSSCREEN_H
#define SETTINGSSCREEN_H

#include "env/config.h"
#include <QMainWindow>

namespace Ui {
class SettingsScreen;
}

class SettingsScreen : public QMainWindow
{
    Q_OBJECT

public:
    explicit SettingsScreen(QWidget *parent = nullptr);
    ~SettingsScreen();

    void fillComboBoxes();

Q_SIGNALS:
    void settingsScreenClosed();

private Q_SLOTS:
    void on_btn_closeWindow_clicked();
    void on_cmb_difficulty_currentTextChanged(const QString &arg1);
    void on_cmb_color_currentTextChanged(const QString &arg1);
    void on_cmb_assist_currentTextChanged(const QString &arg1);

    void on_cmb_assist_activated(int index);

    void on_cmb_color_activated(int index);

    void on_cmb_difficulty_activated(int index);

private:
    Ui::SettingsScreen *ui;
    Config *config;
};

#endif // SETTINGSSCREEN_H
