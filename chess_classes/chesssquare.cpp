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
    if ((rank + file) % 2 == 0) {
        this->setBrush(QColor(110, 110, 102));
    } else {
        this->setBrush(QColor(235, 231, 221));
    }
    return;
}

void ChessSquare::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {           // Right mouse click event
        rightClick();
    } else if (event->button() == Qt::LeftButton) {     // Left mouse click event
        leftClick();
    }
    return;
    // QGraphicsRectItem::mousePressEvent(event);      // Send event to QGraphicsRectItem event handler
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
    if (occupyingPiece == nullptr) {
        // qDebug() << "Right click - No piece on square";
        Q_EMIT squareRightClicked(rank, file);
    } else {
        // qDebug() << "Right click - Piece on square: " << occupyingPiece->getName();
        Q_EMIT squareRightClicked(rank, file);
    }
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
    if (!(this->brush().color() == QColor(252, 223, 116))) {
        this->setBrush(QColor(252, 223, 116));
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

int ChessSquare::getRank()
{
    return rank;
}

int ChessSquare::getFile()
{
    return file;
}
