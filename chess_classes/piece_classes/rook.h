//
// Program Name:              rook.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Rook class. Responsible for:
//

#ifndef ROOK_H
#define ROOK_H

#include "chess_classes/chesspiece.h"

class Rook : public ChessPiece {

public:
    Rook(bool playerIsWhite);
    bool playerIsWhite;
    bool moved = false;
};

#endif // ROOK_H
