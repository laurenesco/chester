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
}

void ChessSquare::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    if (event->button() == Qt::RightButton) {           // Right mouse click event
        leftClick();
    } else if (event->button() == Qt::LeftButton) {     // Left mouse click event
        rightClick();
    }

    QGraphicsRectItem::mousePressEvent(event);      // Send event to QGraphicsRectItem event handler
}

// On right click
void ChessSquare::rightClick()
{
    if (occupyingPiece == nullptr) {
        qDebug() << "No piece on square";
    } else {
        qDebug() << "Piece on square: " << occupyingPiece->getName();
    }
    highlightSquareRed();
}

// On left click
void ChessSquare::leftClick()
{
    highlightSquareYellow();
    highlightPossibleMoves();
}

void ChessSquare::highlightPossibleMoves() {
    return;
}

void ChessSquare::highlightSquareRed() {
    if (!(this->brush().color() == QColor(129, 65, 65))) {
        this->setBrush(QColor(129, 65, 65));
    } else {
        setBaseColor(rank, file);
    }
}

void ChessSquare::highlightSquareYellow() {
    if (!(this->brush().color() == QColor(252, 223, 116))) {
        this->setBrush(QColor(252, 223, 116));
    } else {
        setBaseColor(rank, file);
    }
}

/* ------------- Get and set methods -------------- */

ChessPiece *ChessSquare::getOccupyingPiece() const
{
    return occupyingPiece;
}

void ChessSquare::setOccupyingPiece(ChessPiece *newOccupyingPiece)
{
    occupyingPiece = newOccupyingPiece;
}
