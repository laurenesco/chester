//
// Program Name:              chessboard.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessBoard class. Responsible for:
//

#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include "piece_classes//bishop.h"
#include "piece_classes//rook.h"
#include "piece_classes//pawn.h"
#include "piece_classes//king.h"
#include "piece_classes//queen.h"
#include "piece_classes//knight.h"
#include "chesssquare.h"
#include "chessmove.h"

#include <QWidget>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QPixmap>
#include <QMessageBox>

class ChessBoard : public QWidget {
    Q_OBJECT

public:
    explicit ChessBoard(QWidget* parent = nullptr);

private:
    // Chess squares comprising board
    ChessSquare* boardSquares[8][8];
    int tileSize = 620/8;
    std:: vector<ChessSquare*> highlightedSquares;
    std::vector<ChessSquare*> possibleMoveSquares;
    ChessSquare *selectedSquare = nullptr;
    QGraphicsScene* chessScene;
    QGraphicsView* chessView;
    bool whiteToPlay = true;
    bool movePending = false;
    QString lastMove;
    ChessPiece *lastMovedPiece = nullptr;
    ChessSquare *enPassantSquare = nullptr;
    int halfMoveCounter;
    int fullMoveCounter;
    int fiftyMoveCounter;

    // Castling relevant pieces
    King *whiteKing = nullptr;
    King *blackKing = nullptr;
    Rook *whiteQueensideRook = nullptr;
    Rook *whiteKingsideRook = nullptr;
    Rook *blackQueensideRook = nullptr;
    Rook *blackKingsideRook = nullptr;

    void loadStartingPosition();
    void createChessBoard();
    void addPieceToSquare(ChessPiece *piece, int rank, int file, int color);
    void removePieceFromSquare(ChessSquare *square);
    void onSquareClicked(QGraphicsSceneMouseEvent* event);
    void checkEnPassant();

    ChessSquare* getSquare(int rank, int file);
    void resetHighlightedSquares();
    void resetPossibleMoveSquares();
    bool squareInPossibleMoves(ChessSquare *square);
    void movePiece(ChessSquare *square);
    void highlightPossibleSquares(ChessSquare *square);

    void selectPiece(ChessPiece *selectedPiece, ChessSquare *square);
    void deselectPiece(ChessPiece *selectedPiece, ChessSquare *square);
    void selectSquare(ChessSquare *squareClicked);
    void deselectSquare(ChessSquare *squareClicked);
    void printMoveDebug(QString header);
    QString moveToAlgebraic(ChessPiece *piece, ChessSquare *square);

    QString boardToUCI();
private Q_SLOTS:
    void squareLeftClicked(int rank, int file);
};

#endif // CHESSBOARD_H
