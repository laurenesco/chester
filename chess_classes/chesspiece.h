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
#include <QString>

class ChessPiece
{
public:
    ChessPiece();

    virtual QPixmap getIcon();
    virtual QPixmap getLightIcon();
    virtual QPixmap getDarkIcon();
    virtual void selectPiece();
    virtual int getPossibleMoves(); // Will be pure virutal function
    virtual QString getName();

protected:
    QPixmap icon;
    bool isSelected;
    QString m_name;

private:
};

#endif // CHESSPIECE_H
