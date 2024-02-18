//
// Program Name:              bishop.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Bishop class. See header file for details.
//

#include "bishop.h"

Bishop::Bishop()
{
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//bishop1.png");
    name = "Bishop";
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
        -7, -7
    };
}
