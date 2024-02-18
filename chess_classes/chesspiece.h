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
#include <QDebug>

class ChessPiece
{
public:
    ChessPiece();

    virtual QPixmap getIcon();
    virtual QPixmap getLightIcon();
    virtual QPixmap getDarkIcon();
    virtual QString getName();
    virtual bool getWhite();
    virtual std:: vector<int> getMovesVector();
    virtual bool getIsSelected();

    virtual void setIsSelected(bool);
    virtual void setWhite(bool);

protected:
    QPixmap icon;
    bool pieceSelected;
    QString name;
    std:: vector<int> movesVector;
    bool isWhite;

private:
};

#endif // CHESSPIECE_H
