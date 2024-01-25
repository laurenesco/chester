#ifndef HELPSCREEN_H
#define HELPSCREEN_H

#include <QMainWindow>

namespace Ui {
class HelpScreen;
}

class HelpScreen : public QMainWindow
{
    Q_OBJECT

public:
    explicit HelpScreen(QWidget *parent = nullptr);
    ~HelpScreen();

signals:
    void helpScreenClosed();

private slots:
    void on_btn_closeWindow_clicked();

private:
    Ui::HelpScreen *ui;
};

#endif // HELPSCREEN_H
