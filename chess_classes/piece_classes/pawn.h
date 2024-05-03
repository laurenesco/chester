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
    Pawn(bool playerIsWhite);

    bool playerIsWhite;

    int getMoveCounter();
    void incrementMoveCounter();
    std:: vector<int> getMovesVector();

private:
    int moveCounter = 0;
};

#endif // PAWN_H
