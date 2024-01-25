#ifndef SETTINGSSCREEN_H
#define SETTINGSSCREEN_H

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

signals:
    void settingsScreenClosed();

private slots:
    void on_btn_closeWindow_clicked();

private:
    Ui::SettingsScreen *ui;
};

#endif // SETTINGSSCREEN_H
