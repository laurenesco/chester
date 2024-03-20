//
// Program Name:             pawn.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Pawn class. See header file for details.
//

#include "pawn.h"

Pawn::Pawn()
{
    icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//pawn1.png");
    name = "Pawn";
    movesVector = {
        0, 1
    };
}

bool Pawn::getFirstMove() const
{
    return firstMove;
}

void Pawn::setFirstMove(bool newFirstMove)
{
    firstMove = newFirstMove;
}

std::vector<int> Pawn::getMovesVector()
{
    if (firstMove == true) {
        return { 0, 1, 0, 2 };
    } else {
        return movesVector;
    }
}
