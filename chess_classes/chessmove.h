//
// Program Name:              chesspiece.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessPiece class. Responsible for:
//                                           - Moving the piece on the board
//                                           - Processing the move (see notes in .cpp file for processMove())
//                                           - Storing the move in alegraic notation
//                                           - Saving the move in the database
//

#ifndef CHESSMOVE_H
#define CHESSMOVE_H

#include "chesssquare.h"
#include "chesspiece.h"

#include <QString>

class ChessMove
{
public:
    ChessMove();

private:
    QString algebraicNotation;
    ChessSquare *startSquare;
    ChessSquare *endSquare;
    ChessPiece *pieceMoved;
    ChessPiece *pieceCaptured;

    QString sanityCheck();
    int saveMove();
};

#endif // CHESSMOVE_H
