#ifndef LOGIC_H
#define LOGIC_H

#include "chesspiece.h"
#include "chesssquare.h"

class Logic {

    friend class ChessSquare;

public:
    Logic();

private:
    bool isValidMove(ChessSquare *square, ChessPiece piece);
};

#endif // LOGIC_H
