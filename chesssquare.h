#ifndef CHESSSQUARE_H
#define CHESSSQUARE_H

#include <QGraphicsRectItem>
#include <QBrush>
#include <QGraphicsSceneMouseEvent>

class ChessBoard;

class ChessSquare : public QGraphicsRectItem
{
public:
    ChessSquare(int posX, int posY, int width, int height);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    void highlightSquareRed(ChessSquare *square);
};

#endif // CHESSSQUARE_H
