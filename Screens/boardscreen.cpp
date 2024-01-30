//
// Program Name:              boardscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the BoardScreen class. See header file for details.
//

#include "ui_boardscreen.h"
#include "boardscreen.h"
#include "ChessClasses/chessboard.h"

BoardScreen::BoardScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::BoardScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);

    /*
    // Set styling
    ui->centralwidget->setStyleSheet("background-color: #25292b");
    ui->statusbar->setStyleSheet("background-color: #25292b");
    ui->frmEngine->setStyleSheet("background-color: #25292b");
    ui->frmStats->setStyleSheet("background-color: #353a3d");
    ui->frmMoves->setStyleSheet("background-color: #353a3d");
    */

    // Creating and adding the chessboard to the window
    ChessBoard *chessboard = new ChessBoard(ui->frmBoard);
    chessboard->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

BoardScreen::~BoardScreen()
{
    delete ui;
}

void BoardScreen::on_btn_closeWindow_clicked()
{
    emit boardScreenClosed();
    this->close();
}

