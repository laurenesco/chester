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
    int rank;
    int file;

    void highlightSquareRed(ChessSquare *square);
    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H
