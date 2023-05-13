#ifndef QUEEN_H
#define QUEEN_H

#include "chesspiece.h"

class Queen : public ChessPiece {

public:
    Queen();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // QUEEN_H
