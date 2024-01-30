//
// Program Name:              pawn.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Pawn class. Responsible for:
//

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
