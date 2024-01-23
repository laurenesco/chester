#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QMainWindow>
#include <QLabel>

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
    void on_child_closed(); // Reopen the main window when a child window is closed

private:
    Ui::StartScreen *ui;
};

#endif // STARTSCREEN_H
