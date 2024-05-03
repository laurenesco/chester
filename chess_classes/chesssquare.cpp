//
// Program Name:              chesssquare.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessSquare class. See header file for details.
//

#include "chesssquare.h"

ChessSquare::ChessSquare(int posX, int posY, int width, int height)
    : QGraphicsRectItem(posX, posY, width, height)
{
    this->rank = posX;
    this->file = posY;
    setBaseColor(posX, posY);
}

void ChessSquare::setBaseColor(int rank, int file) {
    if ((rank + file) % 2 != 0) {
        this->setBrush(QColor(52,58,64));
    } else {
        this->setBrush(QColor(206,212,218));
    }
    return;
}

QColor ChessSquare::getBaseColor() {
    if ((rank + file) % 2 != 0) {
        return QColor(52,58,64);
    } else {
        return QColor(206,212,218);
    }
}

void ChessSquare::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (disabled == true) {
        event->ignore();
    } else {
        if (event->button() == Qt::RightButton) {           // Right mouse click event
            rightClick();
        } else if (event->button() == Qt::LeftButton) {     // Left mouse click event
            leftClick();
        }
    }
    return;
}

// On left "main" click
void ChessSquare::leftClick()
{
    Q_EMIT squareLeftClicked(rank, file);
    return;
}

// On right click
void ChessSquare::rightClick()
{
    Q_EMIT squareRightClicked(rank, file);
    return;
}

void ChessSquare::toggleSquareRed() {
    if (!(this->brush().color() == QColor(129, 65, 65))) {
        this->setBrush(QColor(129, 65, 65));
    } else {
        setBaseColor(rank, file);
    }
    return;
}

void ChessSquare::toggleSquareYellow() {
    // Check to see if it is already yellow
    if (this->brush().color() == this->getBaseColor()) {
        QColor whiteColor = QColor(206,212,218);
        if (this->getBaseColor() == whiteColor) {
            // Light sqaures
            this->setBrush(QColor(255,240,173));
        } else {
            // Dark squares
            this->setBrush(QColor(252,226,108));
        }
    } else {
        setBaseColor(rank, file);
    }
    return;
}

void ChessSquare::resetColor()
{
    setBaseColor(rank, file);
    return;
}

void ChessSquare::toggleSquareCustom(QColor color)
{
    this->setBrush(color);
    return;
}

/* ------------- Get and set methods -------------- */

ChessPiece *ChessSquare::getOccupyingPiece() const
{
    return occupyingPiece;
}

void ChessSquare::setOccupyingPiece(ChessPiece *newOccupyingPiece)
{
    occupyingPiece = newOccupyingPiece;
    return;
}

void ChessSquare::setRank(int rank)
{
    if (rank >= 0 && rank < 8) {
        this->rank = rank;
    }
    return;
}

void ChessSquare::setFile(int file)
{
    if (file >= 0 && file < 8) {
        this->file = file;
    }
    return;
}

void ChessSquare::setDisabled(bool newDisabled)
{
    disabled = newDisabled;
}

int ChessSquare::getRank()
{
    return rank;
}

int ChessSquare::getFile()
{
    return file;
}
