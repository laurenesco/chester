#ifndef SPRITESH
#define SPRITESH

#include <QDebug>
#include <QPixmap>

// Functions in the Sprite class
//-------------------------------
// QPixmap getPawn()                 - returns a QPixmap of the .png image for the respective piece type
// QPixmap getBishop()               - ""
// QPixmap getRook()                 - ""
// QPixmap getKnight()               - ""
// QPixmap getKing()                  - ""
// QPixmap getQueen()               - ""

class Sprite
{
private:
    QPixmap pawn;
    QPixmap rook;
    QPixmap knight;
    QPixmap bishop;
    QPixmap king;
    QPixmap queen;

public:
    Sprite();

    QPixmap getPawn();
    QPixmap getRook();
    QPixmap getKnight();
    QPixmap getBishop();
    QPixmap getKing();
    QPixmap getQueen();

};

#endif // SPRITES_H
