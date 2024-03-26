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
#include <QGraphicsPixmapItem>

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
    virtual bool getIsSelected();
    virtual QGraphicsPixmapItem* getSprite();
    QPixmap getSelectedIcon();
    QString getFEN();

    virtual void setIsSelected(bool status);
    virtual void setSprite(QGraphicsPixmapItem *sprite);
    virtual void setColor(int color);

    bool getWhite() const;
    void setWhite(bool newWhite);

protected:
    QPixmap icon;
    QGraphicsPixmapItem *sprite;
    bool pieceSelected;
    QString name;
    std:: vector<int> movesVector;
    int color; // 1 - white, 2 - black, 3 - selected
    bool white;

};

#endif // CHESSPIECE_H
