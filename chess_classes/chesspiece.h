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
    virtual QString getName();
    virtual int getColor();
    virtual std:: vector<int> getMovesVector();

    virtual void setColor(int value);

    virtual void selectPiece();

protected:
    QPixmap icon;
    bool isSelected;
    QString name;
    std:: vector<int> movesVector;
    int isWhite;

private:
};

#endif // CHESSPIECE_H
