//
// Program Name:              boardscreen.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the BoardScreen class. This class is responsible for:
//                                              - Generating the main landing page, or "start screen" of the application
//

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

Q_SIGNALS:
    void boardScreenClosed();

private Q_SLOTS:
    void on_btn_closeWindow_clicked();

private:
    Ui::BoardScreen *ui;
    QWidget *parentForm;;
};

#endif // BOARDSCREEN_H
