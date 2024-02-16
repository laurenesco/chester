//
// Program Name:              chessboard.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessBoard class. See header file for details.
//

#include "chessboard.h"
#include "logic.h"
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

// The 64 chess board ChessSquare objects created here
void ChessBoard::createChessBoard() {
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            ChessSquare *square = new ChessSquare(file * tileSize, rank * tileSize, tileSize, tileSize);
            square->setPen(Qt::NoPen);
            chessScene->addItem(square);

             // Set each entry in the array to a ChessSquare
            boardSquares[rank][file] = square;
            square->setRank(rank);
            square->setFile(file);

            connect(square, &ChessSquare::squareLeftClicked, this, &ChessBoard::highlightPossibleSquares);
            // connect(square, &ChessSquare::squareRightClicked, this, &ChessBoard::highlightPossibleSquares);
            // add this on right click     highlightSquareRed();
        }
    }
}

// Creates all the ChessPiece objects
void ChessBoard::loadStartingPosition() {
    // Pawns
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;
            bool isDark = false;

            Pawn *pawn = new Pawn();

            if (i == 0) {
                lightPawn[j] = pawn;
                pawn->setColor(1);
            } else {
                darkPawn[j] = pawn;
                isDark = true;
                pawn->setColor(0);
            }

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

            if (i == 0) {
                lightRook = rook;
                rook->setColor(1);
            } else {
                darkRook = rook;
                isDark = true;
                rook->setColor(0);
            }

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

            if (i == 0) {
                lightKnight = knight;
                knight->setColor(1);
            } else {
                darkKnight = knight;
                isDark = true;
                knight->setColor(0);
            }

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

            if (i == 0) {
                lightBishop = bishop;
                bishop->setColor(1);
            } else {
                darkBishop = bishop;
                isDark = true;
                bishop->setColor(0);
            }

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

            if (i == 0) {
                lightKing = king;
                king->setColor(1);
            } else {
                darkKing = king;
                isDark = true;
                king->setColor(0);
            }

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

            if (i == 0) {
                lightQueen = queen;
                queen->setColor(1);
            }
            else {
                darkQueen = queen;
                isDark = true;
                queen->setColor(0);
            }

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
    ChessSquare *squares = boardSquares[rank][file];
    QPixmap scaledPiece = pieceSprite.scaled(tileSize-shrinkX, tileSize-shrinkY, Qt::KeepAspectRatio);
    QGraphicsPixmapItem *finalSprite = new QGraphicsPixmapItem(scaledPiece);

    // Position and add to scene
    finalSprite->setPos(squares->rect().topLeft() + QPointF(offsetX, offsetY));
    chessScene->addItem(finalSprite);

    // Add to associated ChessSquare object
    ChessSquare *square = boardSquares[rank][file];
    square->setOccupyingPiece(piece);
    square->setIsOccupied(1);
}

ChessSquare* ChessBoard::getSquare(int rank, int file)
{
    return boardSquares[rank][file];
}

// This function runs if the user left clicks on a square with a piece occupying it
 void ChessBoard::highlightPossibleSquares(int rank, int file) {
     resetHighlightedSquares();
     ChessSquare *square = this->getSquare(rank, file);

     // If the square is not already selected
     if (selectedSquare != square) {
         if (selectedSquare != nullptr) { selectedSquare->resetColor(); }
         selectedSquare = square;
         square->toggleSquareYellow();

        ChessPiece *selectedPiece = boardSquares[rank][file]->getOccupyingPiece();
        std::vector<int> coords = selectedPiece->getMovesVector();

        qDebug() << "Rank: " << rank << " File: " << file;

        // Highlight potential moves yellow
        for (int i = 0; i < (int) coords.size(); i+=2)
        {
            int newRank, newFile;
            int x = coords[i];
            int y = coords[i+1];
            // qDebug() << "coords[i]: " << x << "coords[i+1] " << y;

            if (selectedPiece->getColor()) {
                // Light pieces [Currently opponent side]
                newFile = file - x;          // Change in x-axis
                newRank = rank + y;     // Change in y-axis
            } else {
                // Dark pieces [Currently player side]
                newFile = file + x;          // Change in x-axis
                newRank = rank - y;     // Change in y-axis
            }

            qDebug() << "new rank: " << newRank << "new file:" << newFile;

            // Only highlight the square if it is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                // Do not highlight squares with friendly pieces
                // if (square->getIsOccupied() && !(selectedPiece->getColor() == 1 && boardSquares[newRank][newFile]->getOccupyingPiece()->getColor() == 1)) {
                    ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    if (possibleMove->getIsOccupied()) {
                        qDebug() << "Rank: " << newRank << " File: " << newFile << " " << "IsOccupied";
                        qDebug() << "Piece color: " << selectedPiece->getColor() << " & target square piece color: " << possibleMove->getOccupyingPiece()->getColor();
                    }
                // }
            }
         }
     } else {
         // If square was already actively selected
         square->resetColor();
         selectedSquare = nullptr;
     }

    return;
}

 void ChessBoard::resetHighlightedSquares() {
     while (!highlightedSquares.empty()) {
         ChessSquare *square = highlightedSquares.back();
         highlightedSquares.pop_back();
         square->resetColor();
     }
 }
