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

#include "env/config.h"
#include <QMainWindow>

namespace Ui {
class BoardScreen;
}

class BoardScreen : public QMainWindow
{
    Q_OBJECT

public:
    explicit BoardScreen(Config *config, QWidget *parent = nullptr);
    ~BoardScreen();
    void setMovesLabel(QString updatedString);
    QString getMovesLabel();

Q_SIGNALS:
    void boardScreenClosed();

private Q_SLOTS:
    void on_btn_closeWindow_clicked();
    void moveCompleted(QString algebraic);

private:
    Ui::BoardScreen *ui;
    QWidget *parentForm;
    Config *config;
};

#endif // BOARDSCREEN_H
