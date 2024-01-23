#ifndef PAWN_H
#define PAWN_H

#include "ChessClasses/chesspiece.h"

class Pawn : public ChessPiece {

public:
    Pawn();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // PAWN_H
