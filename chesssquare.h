#ifndef CHESSSQUARE_H
#define CHESSSQUARE_H

#include <QGraphicsRectItem>
#include <QBrush>
#include <QGraphicsSceneMouseEvent>

#include "chesspiece.h"

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
    void preMove(ChessSquare *square);
    bool isValidMove(ChessSquare *square, ChessPiece piece);
};

#endif // CHESSSQUARE_H
