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

    // Create intermediary layout since frmBoard does not have a layout, crashing app when
    // adding the chessboard directly into it. Add frmBoard and frmMoves inside of the new widget
    QWidget *intermediary = new QWidget(ui->frmGame);
    intermediary->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    QVBoxLayout *layout = new QVBoxLayout(intermediary);
    ui->frmBoard->setLayout(layout); // Set the layout for ui->frmBoard

    // Do the same thing for other frames: frmStats and frmEngine
    QVBoxLayout *statsLayout = new QVBoxLayout(ui->frmStats);
    ui->frmStats->setLayout(statsLayout);

    QVBoxLayout *engineLayout = new QVBoxLayout(ui->frmEngine);
    ui->frmEngine->setLayout(engineLayout);

    // Creating and adding the chessboard to the window
    ChessBoard *chessboard = new ChessBoard(ui->frmBoard);
    chessboard->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    layout->addWidget(chessboard); // Add the chessboard to the layout

    // Add the intermediary widget to the frmGame layout
    ui->frmGame->layout()->addWidget(intermediary);
}

MainWindow::~MainWindow()
{
    delete ui;
}

