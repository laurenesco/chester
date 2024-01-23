#ifndef BOARDSCREEN_H
#define BOARDSCREEN_H

#include <QMainWindow>

namespace Ui {
class BoardScreen;
}

class BoardScreen : public QMainWindow
{
    Q_OBJECT

public:
    explicit BoardScreen(QWidget *parent = nullptr);
    ~BoardScreen();

signals:
    void reopen_main_window(); // Send signal the window is closing

private:
    Ui::BoardScreen *ui;
};

#endif // BOARDSCREEN_H
