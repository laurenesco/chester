#ifndef CHESSSQUARE_H
#define CHESSSQUARE_H

#include <QGraphicsRectItem>
#include <QBrush>
#include <QGraphicsSceneMouseEvent>

#include "chesspiece.h"

class ChessBoard;
class Logic;

class ChessSquare : public QGraphicsRectItem {

public:
    ChessSquare(int posX, int posY, int width, int height);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    int rank;
    int file;

    void rightClick(ChessSquare *square);
    void leftClick(ChessSquare *square);

    void highlightSquareRed(ChessSquare *square);
    void highlightSquareYellow(ChessSquare *square);
    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H
