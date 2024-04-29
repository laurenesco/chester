//
// Program Name:              boardscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the BoardScreen class. See header file for details.
//

#include <QLabel>
#include "ui_boardscreen.h"
#include "boardscreen.h"
#include "chess_classes/chessboard.h"
#include "chess_classes/chesssquare.h"
#include "pythoninterface.h"

BoardScreen::BoardScreen(Config *config, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::BoardScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);
    this->config = config;

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

    // Configuring the moves frame
    ui->lbl_moveDisplay->setWordWrap(true);  // Enable word wrap

    connect(chessboard, &ChessBoard::moveCompleted, this, &BoardScreen::moveCompleted);

//    PythonInterface *python = new PythonInterface();
//    python->testPython(this->ui->lbl_eval);
    // Py_SetProgramName(L"C:/Users/laesc/OneDrive/Desktop/chester/python");
}

BoardScreen::~BoardScreen()
{
    delete ui;
}

QString BoardScreen::getMovesLabel()
{
    return ui->lbl_moveDisplay->text();
}

void BoardScreen::setMovesLabel(QString updatedString)
{
    ui->lbl_moveDisplay->setText(updatedString);
    return;
}

void BoardScreen::on_btn_closeWindow_clicked()
{
    Q_EMIT boardScreenClosed();
    this->close();
}

void BoardScreen::moveCompleted(QString algebraic)
{
    if (this->ui->lbl_moveDisplay->text().isEmpty()) {
        this->setMovesLabel(algebraic);
        return;
    }
    QString currentText = this->getMovesLabel();
    currentText = currentText + ", " + algebraic;
    this->setMovesLabel(currentText);
    return;
}

