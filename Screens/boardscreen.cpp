#include "boardscreen.h"
#include "ChessClasses/chessboard.h"
#include "ui_boardscreen.h"

BoardScreen::BoardScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::BoardScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);

    // Set colors
    ui->centralwidget->setStyleSheet("background-color: #25292b");
    ui->statusbar->setStyleSheet("background-color: #25292b");
    ui->frmEngine->setStyleSheet("background-color: #25292b");
    ui->frmStats->setStyleSheet("background-color: #353a3d");
    ui->frmMoves->setStyleSheet("background-color: #353a3d");

    // Creating and adding the chessboard to the window
    ChessBoard *chessboard = new ChessBoard(ui->frmBoard);
    chessboard->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

//void BoardScreen::reopen_main_window()
//{
//    emit reopen_main_window();
//}

BoardScreen::~BoardScreen()
{
    this->reopen_main_window();
    delete ui;
}
