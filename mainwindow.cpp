#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "chessboard.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("Chester");

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

MainWindow::~MainWindow()
{
    delete ui;
}

