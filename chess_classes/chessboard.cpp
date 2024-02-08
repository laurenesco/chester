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

            Pawn *pawn = new Pawn();

            if (i == 0) { lightPawn[j] = pawn; }
            else { darkPawn[j] = pawn; isDark = true; }

            addPieceToOpeningSquare(pawn, 5, 5, 10, 10, rank, j, isDark);
        }
    }

    // Rooks
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;
            bool isDark = false;

            Rook *rook = new Rook();

            if (i == 0) { lightRook = rook; }
            else { darkRook = rook; isDark = true; }

            addPieceToOpeningSquare(rook, 5, 5, 10, 10, rank, file, isDark);
        }
    }

//  Knights
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;
            bool isDark = false;

            Knight *knight = new Knight();

            if (i == 0) { lightKnight = knight; }
            else { darkKnight = knight; isDark = true; }

            addPieceToOpeningSquare(knight, 5, 5, 10, 10, rank, file, isDark);
        }
    }

    // Bishops
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 2 : 5;
            bool isDark = false;

            Bishop *bishop = new Bishop();

            if (i == 0) { lightBishop = bishop; }
            else { darkBishop = bishop; isDark = true; }

            addPieceToOpeningSquare(bishop, 5, 5, 10, 10, rank, file, isDark);
        }
    }

    // Kings
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;
            bool isDark = false;

            King *king = new King();

            if (i == 0) { lightKing = king; }
            else { darkKing = king; isDark = true; }

            addPieceToOpeningSquare(king, 5, 5, 10, 10, rank, file, isDark);
          }
    }

    // Queens
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;
            bool isDark = false;

            Queen *queen = new Queen();

            if (i == 0) { lightQueen = queen; }
            else { darkQueen = queen; isDark = true; }

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
