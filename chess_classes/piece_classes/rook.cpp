//
// Program Name:              rook.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Rook class. See header file for details.
//

#include "rook.h"

Rook::Rook()
{
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//rook1.png");
    name = "Rook";
    movesVector = {
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
        0, 0,
        7, 0,
        -1, 0,
        -2, 0,
        -3, 0,
        -4, 0,
        -5, 0,
        -6, 0,
        -7, 0,
    };
}
