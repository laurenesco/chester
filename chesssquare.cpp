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
    if (event->button() == Qt::RightButton) {
        highlightSquareRed(this);
    } else if (event->button() == Qt::LeftButton) {

    }
    QGraphicsRectItem::mousePressEvent(event);
}

void ChessSquare::highlightSquareRed(ChessSquare *square) {
    if (!(square->brush().color() == QColor(129, 65, 65))) {
        square->setBrush(QColor(129, 65, 65));
    } else {
        setBaseColor(rank, file);
    }
}
