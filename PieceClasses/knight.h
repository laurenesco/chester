#ifndef KNIGHT_H
#define KNIGHT_H

#include "ChessClasses/chesspiece.h"

class Knight : public ChessPiece {

public:
    Knight();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // KNIGHT_H
