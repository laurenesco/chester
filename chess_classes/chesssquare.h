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

#include <QDebug>

// #include "chessboard.h"
#include "chesspiece.h"

class ChessBoard;

class ChessSquare : public QObject, public QGraphicsRectItem {
    Q_OBJECT

public:
    ChessSquare(int posX, int posY, int width, int height);

    // Get/set methods
    ChessPiece *getOccupyingPiece() const;
    void setOccupyingPiece(ChessPiece *newOccupyingPiece);
    void setRank(int rank);
    void setFile(int file);
    void highlightSquareRed();
    void highlightSquareYellow();

Q_SIGNALS:
    void squareClicked(int rank, int file);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    int rank;
    int file;
    ChessPiece *occupyingPiece = nullptr;

    void rightClick();
    void leftClick();

   //  void highlightPossibleMoves();
    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H
