#include "startscreen.h"
#include "ui_startscreen.h"

StartScreen::StartScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StartScreen)
{
    ui->setupUi(this);
    ui->centralwidget->setStyleSheet("background-color: #25292b");

    QString path = R"(C:/Users/laesc/OneDrive/Desktop/chester/logos/chesster.png)";
    QPixmap img(path);
    ui->lbl_logo->setPixmap(img);
}

StartScreen::~StartScreen()
{
    delete ui;
}
