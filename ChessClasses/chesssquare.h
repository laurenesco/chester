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

    void rightClick(ChessSquare *square);
    void leftClick(ChessSquare *square);

    void highlightSquareRed(ChessSquare *square);
    void highlightSquareYellow(ChessSquare *square);
    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H