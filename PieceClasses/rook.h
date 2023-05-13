#ifndef ROOK_H
#define ROOK_H

#include "chesspiece.h"

class Rook : public ChessPiece {

public:
    Rook();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // ROOK_H
