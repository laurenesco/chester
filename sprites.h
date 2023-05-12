#ifndef SPRITES_H
#define SPRITES_H

#include <QDebug>
#include <QPixmap>

class sprites
{
private:
    QPixmap pawn;
    QPixmap rook;
    QPixmap knight;
    QPixmap bishop;
    QPixmap king;
    QPixmap queen;

public:
    sprites();
    QPixmap getPawn();
    QPixmap getRook();
    QPixmap getKnight();
    QPixmap getBishop();
    QPixmap getKing();
    QPixmap getQueen();

};

#endif // SPRITES_H
