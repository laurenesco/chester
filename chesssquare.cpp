#include "chesssquare.h"

ChessSquare::ChessSquare(int posX, int posY, int width, int height)
    : QGraphicsRectItem(posX, posY, width, height)
{
    // Set the alternating square colors
    if ((posX + posY) % 2 == 0) {
        setBrush(QColor(110, 110, 102));
    } else {
        setBrush(QColor(235, 231, 221));
    }
}

void ChessSquare::mousePressEvent(QGraphicsSceneMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        highlightSquareRed(this);
    } else if (event->button() == Qt::RightButton) {

    }

    QGraphicsRectItem::mousePressEvent(event);
}

void ChessSquare::highlightSquareRed(ChessSquare *square) {
    if (!(square->brush().color() == QColor(129, 65, 65))) {
        square->setBrush(QColor(129, 65, 65));
    }
}
