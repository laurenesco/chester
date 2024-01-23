#ifndef BISHOP_H
#define BISHOP_H

#include "ChessClasses/chesspiece.h"
#include <QPixmap>

class Bishop : public ChessPiece {

public:
    Bishop();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // BISHOP_H
