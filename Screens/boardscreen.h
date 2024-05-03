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
#include "evaluationbar.h"
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
    void setMovesLabel(QString updatedString);
    QString getMovesLabel();

Q_SIGNALS:
    void boardScreenClosed();
    void blackToPlay();

private:
    Ui::BoardScreen *ui;
    QWidget *parentForm;
    Config *config;
    EvaluationBar *evalBar;
    void setMascot();

    struct evaluation {
        double value = 0; // Represents advantage if cp, or moves until mate if mate
        int winning = 0;  // 1 - white, 2 - black
        int status = 0;   // 1 - cp, 2 - mate
    };

private Q_SLOTS:
    void on_btn_closeWindow_clicked();
    void moveCompleted(QString algebraic, int winning, int evaluation); // See evaluation struct in ChessBoard.h for details on winning and evaluation
    void switchMascot(int status);
    void game_over(QString notification);
};

#endif // BOARDSCREEN_H
