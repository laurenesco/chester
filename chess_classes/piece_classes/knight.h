//
// Program Name:              knight.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Knight class. Responsible for:
//

#ifndef KNIGHT_H
#define KNIGHT_H

#include "chess_classes/chesspiece.h"

class Knight : public ChessPiece {

public:
    Knight(bool playerIsWhite);
    bool playerIsWhite;
};

#endif // KNIGHT_H
