//
// Program Name:              chessboard.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessBoard class. Responsible for:
//

#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include "PieceClasses//bishop.h"
#include "PieceClasses//rook.h"
#include "PieceClasses//pawn.h"
#include "PieceClasses//king.h"
#include "PieceClasses//queen.h"
#include "PieceClasses//knight.h"
#include "chesssquare.h"

#include <QWidget>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsItem>
#include <QPixmap>

class ChessBoard : public QWidget {
    Q_OBJECT

public:
    explicit ChessBoard(QWidget* parent = nullptr);

private:
    QGraphicsScene* chessScene;
    QGraphicsView* chessView;
    QGraphicsRectItem* chessSquares[8][8];
    QPixmap pieceSprite;

    // Light pieces
    Pawn lightPawn[8];
    Rook lightRook;
    Bishop lightBishop;
    Knight lightKnight;
    King lightKing;
    Queen lightQueen;

    // Dark piece
    Pawn darkPawn[8];
    Rook darkRook;
    Bishop darkBishop;
    Knight darkKnight;
    King darkKing;
    Queen darkQueen;

    int tileSize = 620/8;

    void loadStartingPosition();
    void createChessBoard();
    void addPieceToOpeningSquare(ChessPiece *piece, int offsetX, int offsetY, int shrinkX, int shrinkY, int rank, int file, bool isDark);
    void onSquareClicked(QGraphicsSceneMouseEvent* event);
};

#endif // CHESSBOARD_H
