//
// Program Name:              king.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the King class. See header file for details.
//

#include "king.h"

King::King(bool playerIsWhite) : ChessPiece(playerIsWhite)
{
    this->playerIsWhite = playerIsWhite;
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//king1.png");
    name = "King";
    movesVector = {
        -1, 1,
        0, 0,
        0, 1,
        0, 0,
        1, 1,
        0, 0,
        1, 0,
        0, 0,
        1, -1,
        0, 0,
        0, -1,
        0, 0,
        -1, -1,
        0, 0,
        -1, 0
    };
}
