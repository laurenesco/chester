//
// Program Name:              chessboard.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessBoard class. See header file for details.
//

#include "chessboard.h"
#include "qboxlayout.h"

ChessBoard::ChessBoard(QWidget* parent) : QWidget(parent) {
    chessScene = new QGraphicsScene(this);
    chessScene->setBackgroundBrush(Qt::transparent);

    chessView = new QGraphicsView(chessScene, this);
    chessView->setBackgroundBrush(Qt::transparent);
    chessView->setFrameStyle(QFrame::NoFrame);
    chessView->setFixedSize(700, 700);
    chessView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    chessView->fitInView(chessScene->sceneRect(), Qt::KeepAspectRatio);

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(chessView);
    setLayout(layout);

    createChessBoard();
    loadStartingPosition();
}

void ChessBoard::createChessBoard() {
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            ChessSquare *square = new ChessSquare(file * tileSize, rank * tileSize, tileSize, tileSize);
            square->setPen(Qt::NoPen);
            chessScene->addItem(square);

             // Set each entry in the array to a ChessSquare
            chessSquares[rank][file] = square;
        }
    }
}

void ChessBoard::loadStartingPosition() {
    // Pawns
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;
            bool isDark = false;

            ChessPiece *pawn;

            if (i == 0) { pawn = &lightPawn[j]; }
            else { pawn = &darkPawn[j]; isDark = true; }

            addPieceToOpeningSquare(pawn, 5, 5, 10, 10, rank, j, isDark);
        }
    }

    // Rooks
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;
            bool isDark = false;

            ChessPiece *rook = new Rook();

            if (i == 0) { rook = &lightRook; }
            else { rook = &darkRook; isDark = true; }

            addPieceToOpeningSquare(rook, 5, 5, 10, 10, rank, file, isDark);
        }
    }

//  Knights
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;
            bool isDark = false;

            ChessPiece *knight;

            if (i == 0) { knight = &lightKnight; }
            else { knight = &darkKnight; isDark = true; }

            addPieceToOpeningSquare(knight, 5, 5, 10, 10, rank, file, isDark);
        }
    }

    // Bishops
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 2 : 5;
            bool isDark = false;

            ChessPiece *bishop;

            if (i == 0) { bishop = &lightBishop; }
            else { bishop = &darkBishop; isDark = true; }

            addPieceToOpeningSquare(bishop, 5, 5, 10, 10, rank, file, isDark);
        }
    }

    // Kings
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;
            bool isDark = false;

            ChessPiece *king;

            if (i == 0) { king = &lightKing; }
            else { king = &darkKing; isDark = true; }

            addPieceToOpeningSquare(king, 5, 5, 10, 10, rank, file, isDark);
          }
    }

    // Queens
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;
            bool isDark = false;

            ChessPiece *queen;

            if (i == 0) { queen = &lightQueen; }
            else { queen = &darkQueen; isDark = true; }

            addPieceToOpeningSquare(queen, 5, 5, 10, 10, rank, file, isDark);
          }
    }
}

void ChessBoard::addPieceToOpeningSquare(ChessPiece *piece, int offsetX, int offsetY, int shrinkX, int shrinkY, int rank, int file, bool isDark)
{
    // Create and scale the sprite
    QPixmap pieceSprite;
    if (isDark) { pieceSprite = piece->getDarkIcon(); }
    else { pieceSprite = piece->getLightIcon(); }
    ChessSquare *squares = chessSquares[rank][file];
    QPixmap scaledPiece = pieceSprite.scaled(tileSize-shrinkX, tileSize-shrinkY, Qt::KeepAspectRatio);
    QGraphicsPixmapItem *finalSprite = new QGraphicsPixmapItem(scaledPiece);

    // Position and add to scene
    finalSprite->setPos(squares->rect().topLeft() + QPointF(offsetX, offsetY));
    chessScene->addItem(finalSprite);

    // Add to associated ChessSquare object
    ChessSquare *square = chessSquares[rank][file];
    square->setOccupyingPiece(piece);
}
