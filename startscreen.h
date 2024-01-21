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

private:
    Ui::StartScreen *ui;
};

#endif // STARTSCREEN_H
