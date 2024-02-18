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

            connect(square, &ChessSquare::squareLeftClicked, this, &ChessBoard::squareLeftClicked);
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
                pawn->setWhite(1);
            } else {
                darkPawn[j] = pawn;
                isDark = true;
                pawn->setWhite(0);
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
                rook->setWhite(1);
            } else {
                darkRook = rook;
                isDark = true;
                rook->setWhite(0);
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
                knight->setWhite(1);
            } else {
                darkKnight = knight;
                isDark = true;
                knight->setWhite(0);
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
                bishop->setWhite(1);
            } else {
                darkBishop = bishop;
                isDark = true;
                bishop->setWhite(0);
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
                king->setWhite(1);
            } else {
                darkKing = king;
                isDark = true;
                king->setWhite(0);
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
                queen->setWhite(1);
            }
            else {
                darkQueen = queen;
                isDark = true;
                queen->setWhite(0);
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

// Highlight all potential moves based on the piece selected
void ChessBoard::highlightPossibleSquares(ChessSquare *square) {

    ChessPiece *selectedPiece = square->getOccupyingPiece();
    selectedPiece->setIsSelected(true);
    std::vector<int> coords = selectedPiece->getMovesVector();
    bool lineStopped = false; // Marked true when ecountering a square with friendly piece, since you cannot move through them

    // Highlight potential moves yellow
    for (int i = 0; i < (int) coords.size(); i+=2)
    {
        int newRank, newFile;
        int x = coords[i];
        int y = coords[i+1];

        // Reset lineStopped when a new direction is started (marked by 0, 0 coords)
        if (x == 0 && y == 0) {
            lineStopped = false;
        }

        if (!lineStopped) {
            if (selectedPiece->getWhite()) {
                // Light pieces [Currently opponent side]
                newFile = square->getFile() - x;          // Change in x-axis
                newRank = square->getRank() + y;     // Change in y-axis
            } else {
                // Dark pieces [Currently player side]
                newFile = square->getFile() + x;          // Change in x-axis
                newRank = square->getRank() - y;     // Change in y-axis
            }

            // Only highlight the square if it is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                bool squareOccupied = possibleMove->getIsOccupied() == 1 ? true : false;

                // Only highlight if it has enemy piece or empty
                if (!squareOccupied) {
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                } else if ((selectedPiece->getWhite() && !possibleMove->getOccupyingPiece()->getWhite()) ||
                           (!selectedPiece->getWhite() && possibleMove->getOccupyingPiece()->getWhite())) {
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                } else {
                    lineStopped = true;
                }
            }
        }
    }
    return;
}

void ChessBoard::resetHighlightedSquares() {
    while (!highlightedSquares.empty()) {
        ChessSquare *square = highlightedSquares.back();
        highlightedSquares.pop_back();
        square->resetColor();
    }
    return;
}

void ChessBoard::resetPossibleMoveSquares() {
    while(!possibleMoveSquares.empty()) {
        possibleMoveSquares.pop_back();
    }
    return;
}

bool ChessBoard::squareInPossibleMoves(ChessSquare *square)
{
    for (ChessSquare *s : possibleMoveSquares) {
        if (s == square) {
            return true;
        }
    }
    return false;
}

void ChessBoard::movePiece(ChessSquare *squareActivated)
{
    ChessSquare *originalSquare = this->selectedSquare;
    ChessPiece *pieceToMove = originalSquare->getOccupyingPiece();
    bool isDark = pieceToMove->getWhite() == false ? true : false;
    addPieceToOpeningSquare(pieceToMove, 5, 5, 10, 10, squareActivated->getRank(), squareActivated->getFile(), isDark);
    selectedSquare = nullptr;
    return;
    // get piece on selected square
    // move to target square
    // put selected square to nullptr
}

// This function runs if the user left clicks on a square
void ChessBoard::squareLeftClicked(int rank, int file)
{
    // Check that click was legal
    if (boardSquares[rank][file] == nullptr) {
        qDebug() << "User has somehow clicked on a square that does not exist";
    } else {
        ChessSquare *squareClicked = this->getSquare(rank, file);

        // If move currently pending
        if (!possibleMoveSquares.empty()) {
            if (squareInPossibleMoves(squareClicked)) {
                qDebug() << "Moving piece.";
                movePiece(squareClicked);
            }
        }

        // Reset data from previous click
        resetPossibleMoveSquares();
        resetHighlightedSquares();

        // If the square is not already selected
        if (selectedSquare != squareClicked) {
            qDebug() << "Square was not previously selected.";
            if (selectedSquare != nullptr) {
                selectedSquare->resetColor();
            }
            selectedSquare = squareClicked;
            squareClicked->toggleSquareYellow();
            if (squareClicked->getIsOccupied()) {
                highlightPossibleSquares(squareClicked);
            }
        } else { // Else if the square was already actively selected
            qDebug() << "Square was previously selected.";
            squareClicked->resetColor();
            selectedSquare = nullptr;
        }
    }
    return;
}
