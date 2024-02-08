//
// Program Name:              pawn.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Pawn class. Responsible for:
//

#ifndef PAWN_H
#define PAWN_H

#include "chess_classes/chesspiece.h"

class Pawn : public ChessPiece {

public:
    Pawn();

private:
    QString name = "Pawn";
};

#endif // PAWN_H
