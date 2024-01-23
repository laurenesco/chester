#include "startscreen.h"
#include "boardscreen.h"
#include "ui_startscreen.h"

StartScreen::StartScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::StartScreen)
{
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);

    QString path = R"(C:/Users/laesc/OneDrive/Desktop/chester/logos/chesster.png)";
    QPixmap img(path);
    ui->lbl_logo->setPixmap(img);
}

StartScreen::~StartScreen()
{
    delete ui;
}

void StartScreen::on_btn_play_clicked()
{
    BoardScreen *b = new BoardScreen(this);
    this->hide();
    b->show();
}

