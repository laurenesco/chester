#ifndef BISHOP_H
#define BISHOP_H

#include "chesspiece.h"
#include <QPixmap>

class Bishop : public ChessPiece {

public:
    Bishop();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // BISHOP_H
