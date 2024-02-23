//
// Program Name:              knight.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Knight class. See header file for details.
//

#include "knight.h"

Knight::Knight()
{
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//knight1.png");
    name = "Knight";
    movesVector = {
        -2, 1,
        0, 0,
        -1, 2,
        0, 0,
        1, 2,
        0, 0,
        2, 1,
        0, 0,
        2, -1,
        0, 0,
        1, -2,
        0, 0,
        -1, -2,
        0, 0,
        -2, -1};
}
