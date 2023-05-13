#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include "sprite.h"
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

// Functions in the ChessBoard class
//--------------------------------------
// ChessBoard(QWidget *parent)     - Configures the QGraphicsScene and QGraphicsView where the game takes place
// void createChessBoard()               - Generates the gridLayout and Labels which act as the board
// void loadStartingPosition()           - Generates the correct rank and file positions for all the pieces, then calls addSpriteToScene()
// void addSpriteToScene(...)            - Creates, scales, and adds the sprite to the specified location

class ChessBoard : public QWidget {
    Q_OBJECT

public:
    explicit ChessBoard(QWidget* parent = nullptr);

    bool preMove = false;
    bool postMove = false;

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

    void createChessBoard();
    void loadStartingPosition();
    void addPieceToScene(ChessPiece *piece, int offsetX, int offsetY, int shrinkX, int shrinkY, int rank, int file, bool isDark);
    void onSquareClicked(QGraphicsSceneMouseEvent* event);
};

#endif // CHESSBOARD_H
