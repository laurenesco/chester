//
// Program Name:              bishop.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Bishop class. Responsible for:
//

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
