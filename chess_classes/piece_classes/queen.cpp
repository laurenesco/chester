//
// Program Name:              queen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Queen class. See header file for details.
//

#include "queen.h"

Queen::Queen(bool playerIsWhite) : ChessPiece(playerIsWhite)
{
    this->playerIsWhite = playerIsWhite;
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//queen1.png");
    name = "Queen";
    movesVector = {
        1, -1,
        2, -2,
        3, -3,
        4, -4,
        5, -5,
        6, -6,
        7, -7,
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
        7, 7,
        0, 0,
        -1, 1,
        -2, 2,
        -3, 3,
        -4, 4,
        -5, 5,
        -6, 6,
        -7, 7,
        0, 0,
        -1, -1,
        -2, -2,
        -3, -3,
        -4, -4,
        -5, -5,
        -6, -6,
        -7, -7,
        0, 0,
        0, -1,
        0, -2,
        0, -3,
        0, -4,
        0, -5,
        0, -6,
        0, -7,
        0, 0,
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        0, 5,
        0, 6,
        0, 7,
        0, 0,
        1, 0,
        2, 0,
        3, 0,
        4, 0,
        5, 0,
        6, 0,
        7, 0,
        0, 0,
        -1, 0,
        -2, 0,
        -3, 0,
        -4, 0,
        -5, 0,
        -6, 0,
        -7, 0,
    };
}
