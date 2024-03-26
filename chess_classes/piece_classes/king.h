//
// Program Name:              king.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the King class. Responsible for:
//

#ifndef KING_H
#define KING_H

#include "chess_classes/chesspiece.h"

class King : public ChessPiece {

public:
    King();
    bool moved;
};

#endif // KING_H
