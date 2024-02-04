//
// Program Name:              helpscreen.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the HelpScreen class. This class is responsible for:
//                                              - Generating the help page of the application
//                                              - Displaying the following upon user request:
//                                                      - Rules of chess
//                                                      - Appendix
//

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

Q_SIGNALS:
    void helpScreenClosed();

private Q_SLOTS:
    void on_btn_closeWindow_clicked();

private:
    Ui::HelpScreen *ui;
};

#endif // HELPSCREEN_H
