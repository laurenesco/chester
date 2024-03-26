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

    bool initiateMove(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare* (&boardSquares)[8][8], bool movePending,
                                   std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares, ChessSquare *selectedSquare);


private:
    QString algebraicNotation;
    ChessSquare *startSquare;
    ChessSquare *endSquare;
    ChessPiece *pieceMoved;
    ChessPiece *pieceCaptured;
    int tileSize = 620/8;
    bool movePending;

    void printMoveDebug(QString header);
    void resetPossibleMoveSquares(std::vector<ChessSquare *> &possibleMoveSquares);
    void resetHighlightedSquares(std::vector<ChessSquare*> &highlightedSquares);
    bool squareInPossibleMoves(ChessSquare *square, std::vector<ChessSquare *> &possibleMoveSquares);
    void deselectSquare(ChessSquare * squareClicked, ChessSquare *selectedSquare);
    void selectSquare(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare *selectedSquare, ChessSquare* (&boardSquares)[8][8],
                                  bool movePending, std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares);
    void movePiece(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare *selectedSquare, bool movePending);
    void highlightPossibleSquares(ChessSquare *square, QGraphicsScene *chessScene, ChessSquare* (&boardSquares)[8][8], bool movePending,
                                                         std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares);
    void deselectPiece(ChessPiece *selectedPiece, ChessSquare *square, QGraphicsScene *chessScene);
    void selectPiece(ChessPiece *selectedPiece, ChessSquare *square, QGraphicsScene *chessScene);
    void removePieceFromSquare(ChessSquare *square, QGraphicsScene *chessScene);
    void addPieceToSquare(ChessPiece *piece, ChessSquare *square, int color, QGraphicsScene *chessScene);

    QString sanityCheck();
    int saveMove();
};

#endif // CHESSMOVE_H
