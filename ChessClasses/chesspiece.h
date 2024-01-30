//
// Program Name:              chesspiece.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessPiece class. Responsible for:
//

#ifndef CHESSPIECE_H
#define CHESSPIECE_H

#include <QPixmap>
#include <QPainter>

class ChessPiece
{
public:
    ChessPiece();

    virtual QPixmap getIcon();
    virtual QPixmap getLightIcon();
    virtual QPixmap getDarkIcon();

private:
    QPixmap icon;

    bool isSelected;
};

#endif // CHESSPIECE_H
