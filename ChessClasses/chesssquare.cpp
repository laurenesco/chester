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

void ChessSquare::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::RightButton) {           // Right mouse click event
        rightClick(this);
    } else if (event->button() == Qt::LeftButton) {     // Left mouse click event
        leftClick(this);
    }

    QGraphicsRectItem::mousePressEvent(event);      // Send event to QGraphicsRectItem event handler
}

void ChessSquare::rightClick(ChessSquare *square)
{
    highlightSquareRed(square);
}

void ChessSquare::leftClick(ChessSquare *square)
{
    highlightSquareYellow(square);
}

void ChessSquare::highlightSquareRed(ChessSquare *square) {
    if (!(square->brush().color() == QColor(129, 65, 65))) {
        square->setBrush(QColor(129, 65, 65));
    } else {
        setBaseColor(rank, file);
    }
}

void ChessSquare::highlightSquareYellow(ChessSquare *square) {
    if (!(square->brush().color() == QColor(252, 223, 116))) {
        square->setBrush(QColor(252, 223, 116));
    } else {
        setBaseColor(rank, file);
    }
}
