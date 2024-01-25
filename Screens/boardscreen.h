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
    void boardScreenClosed();

private slots:
    void on_btn_closeWindow_clicked();

private:
    Ui::BoardScreen *ui;
    QWidget *parentForm;;
};

#endif // BOARDSCREEN_H
