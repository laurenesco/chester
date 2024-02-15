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
        rightClick();
    } else if (event->button() == Qt::LeftButton) {     // Left mouse click event
        leftClick();
    }

    QGraphicsRectItem::mousePressEvent(event);      // Send event to QGraphicsRectItem event handler
}

// On left "main" click
void ChessSquare::leftClick()
{
    if (occupyingPiece == nullptr) {
        qDebug() << "Left click - No piece on square";
    } else {
        qDebug() << "Left click - Piece on square: " << occupyingPiece->getName();
        // highlightPossibleMoves();
        Q_EMIT squareClicked(rank, file);
    }
    highlightSquareYellow();
}

// On right click
void ChessSquare::rightClick()
{
    if (occupyingPiece == nullptr) {
        qDebug() << "RIght click - No piece on square";
    } else {
        qDebug() << "Right click - Piece on square: " << occupyingPiece->getName();
    }
    highlightSquareRed();
}

// Highlight possible moves for the selected piece
//void ChessSquare::highlightPossibleMoves() {
//    std::vector<int> coords = occupyingPiece->getMovesVector();

//    qDebug() << "Rank: " << rank << " File: " << file;

//    for (int i = 0; i < (int) coords.size()/2; i ++)
//    {
//        int x = coords[i];
//        int y = coords[i+1];

//        int newX = rank + x;
//        int newY = rank + y;

//        // ChessSquare *possibleSquare = parentBoard->getSquare(newX, newY);
//    }

//    return;
//}

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

void ChessSquare::setRank(int rank)
{
    if (rank >= 0 && rank < 8) {
        this->rank = rank;
    }
}

void ChessSquare::setFile(int file)
{
    if (file >= 0 && file < 8) {
        this->file = file;
    }
}
