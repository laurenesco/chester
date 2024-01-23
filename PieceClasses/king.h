#ifndef KING_H
#define KING_H

#include "ChessClasses/chesspiece.h"

class King : public ChessPiece {

public:
    King();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // KING_H
