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

#include "chesspiece.h"

class ChessSquare : public QObject, public QGraphicsRectItem {
    Q_OBJECT

public:
    ChessSquare(int posX, int posY, int width, int height, bool white);

    // Get/set methods
    ChessPiece *getOccupyingPiece() const;
    void setOccupyingPiece(ChessPiece *newOccupyingPiece);
    void setRank(int rank);
    void setFile(int file);
    bool disabled = false;
    bool playerIsWhite;

    int getRank();
    int getFile();

    void toggleSquareRed();
    void toggleSquareYellow();
    void resetColor();
    void toggleSquareCustom(QColor color);

    QColor getBaseColor();
    void setDisabled(bool newDisabled);

Q_SIGNALS:
    void squareLeftClicked(int rank, int file);
    void squareRightClicked(int rank, int file);

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;

private:
    int rank;
    int file;
    ChessPiece *occupyingPiece = nullptr;

    void rightClick();
    void leftClick();

    void setBaseColor(int rank, int file);
};

#endif // CHESSSQUARE_H
