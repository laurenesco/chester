//
// Program Name:              logic.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Logic class. Responsible for:
//

#ifndef LOGIC_H
#define LOGIC_H

#include "ChessClasses/chesspiece.h"
#include "ChessClasses/chesssquare.h"

class Logic {

    friend class ChessSquare;

public:
    Logic();

private:
    bool isValidMove(ChessSquare *square, ChessPiece piece);
};

#endif // LOGIC_H
