//
// Program Name:              chesssquare.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessSquare class. Responsible for:
//

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
    ChessPiece occupyingPiece;

    void rightClick(ChessSquare *self);
    void leftClick(ChessSquare *self);

    void highlightSquareRed(ChessSquare *self);
    void highlightSquareYellow(ChessSquare *self);
    void highlightPossibleMoves(ChessSquare *self);
    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H
