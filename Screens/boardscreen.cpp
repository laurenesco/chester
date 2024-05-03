//
// Program Name:              boardscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the BoardScreen class. See header file for details.
//

#include <QLabel>
#include <QMovie>
#include "ui_boardscreen.h"
#include "boardscreen.h"
#include "chess_classes/chessboard.h"
#include "chess_classes/chesssquare.h"
#include "pythoninterface.h"

BoardScreen::BoardScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::BoardScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster");
    setGeometry(200, 85, 1500, 900);

    // Create the evaluation bar
    evalBar = new EvaluationBar();
    evalBar->setStyleSheet("border-radius: 10px;"); // Not working :(
    this->ui->frm_eval->layout()->addWidget(evalBar);

    // Creating and adding the chessboard to the window
    ChessBoard *chessboard = new ChessBoard(ui->frmBoard);
    chessboard->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    // Configuring the moves frame
    ui->lbl_moveDisplay->setWordWrap(true);  // Enable word wrap

    // Connect to the moveCompleted signal in ChessBoard class
    connect(chessboard, &ChessBoard::moveCompleted, this, &BoardScreen::moveCompleted);
    connect(chessboard, &ChessBoard::game_over, this, &BoardScreen::game_over);
    connect(chessboard, &ChessBoard::switchMascot, this, &BoardScreen::switchMascot);
}

BoardScreen::~BoardScreen()
{
    delete ui;
}

QString BoardScreen::getMovesLabel()
{
    return ui->lbl_moveDisplay->text();
}

void BoardScreen::setMascot()
{
//    QString path = R"(C:/Users/laesc/OneDrive/Desktop/chester/icons/still_mascot.png)";
//    QPixmap img(path);
//    img = img.scaled(img.size()*.9, Qt::KeepAspectRatio);
//    ui->lbl_mascot->setPixmap(img);
    return;
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

void BoardScreen::switchMascot(int status) {
//    if (status == 1) {
//        QMovie *movie = new QMovie("C:/Users/laesc/OneDrive/Desktop/chester/icons/knight_mirrored.gif");
//        ui->lbl_mascot->setMovie(movie);
//        ui->lbl_mascot->setScaledContents(true);
//        movie->start();
//    } else {
//        setMascot();
//    }

    return;

}

void BoardScreen::game_over(QString notification)
{
    qDebug() << layout()->count();
    QLayoutItem *item = ui->frm_eval->layout()->itemAt(0);
    ui->frm_eval->layout()->removeItem(item);

    QLabel *label = new QLabel();
    label->setText(notification);
    label->setAlignment(Qt::AlignCenter);
    this->ui->frm_eval->layout()->addWidget(label);

    return;

}

void BoardScreen::moveCompleted(QString algebraic, int winning, int evaluation)
{
    // Update move bank
    if (this->ui->lbl_moveDisplay->text().isEmpty()) {
        this->setMovesLabel(algebraic);
        return;
    }

    QString currentText = this->getMovesLabel();
    currentText = currentText + ", " + algebraic;
    this->setMovesLabel(currentText);

    // Update evaluation bar
    int value = winning == 1 ? evaluation : (0 - evaluation);
    evalBar->setEvaluation(value);

    return;
}

