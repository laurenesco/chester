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

/* Constructor */
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

/* The 64 chess board ChessSquare objects created here */
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

/* Creates all the ChessPiece objects */
void ChessBoard::loadStartingPosition() {
    // Pawns
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;

            Pawn *pawn = new Pawn();

            if (i == 0) {
                lightPawn[j] = pawn;
                pawn->setWhite(true);
            } else {
                darkPawn[j] = pawn;
                pawn->setWhite(false);
            }

            int color = pawn->getWhite() == true ? 1 : 2;
            addPieceToSquare(pawn, rank, j, color);
        }
    }

    // Rooks
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;

            Rook *rook = new Rook();

            if (i == 0) {
                lightRook = rook;
                rook->setWhite(true);
            } else {
                darkRook = rook;
                rook->setWhite(false);
            }

            int color = rook->getWhite() == true ? 1 : 2;
            addPieceToSquare(rook, rank, file, color);
        }
    }

    //  Knights
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;

            Knight *knight = new Knight();

            if (i == 0) {
                lightKnight = knight;
                knight->setWhite(true);
            } else {
                darkKnight = knight;
                knight->setWhite(false);
            }

            int color = knight->getWhite() == true ? 1 : 2;
            addPieceToSquare(knight, rank, file, color);
        }
    }

    // Bishops
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 2 : 5;

            Bishop *bishop = new Bishop();

            if (i == 0) {
                lightBishop = bishop;
                bishop->setWhite(true);
            } else {
                darkBishop = bishop;
                bishop->setWhite(false);
            }

            int color = bishop->getWhite() == true ? 1 : 2;
            addPieceToSquare(bishop, rank, file, color);
        }
    }

    // Kings
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;

            King *king = new King();

            if (i == 0) {
                lightKing = king;
                king->setWhite(true);
            } else {
                darkKing = king;
                king->setWhite(false);
            }

            int color = king->getWhite() == true ? 1 : 2;
            addPieceToSquare(king, rank, file, color);
        }
    }

    // Queens
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;

            Queen *queen = new Queen();

            if (i == 0) {
                lightQueen = queen;
                queen->setWhite(true);
            }
            else {
                darkQueen = queen;
                queen->setWhite(false);
            }

            int color = queen->getWhite() == true ? 1 : 2;
            addPieceToSquare(queen, rank, file, color);
        }
    }
}

/* Add a given chess piece to a square specified by rank and file [Color code: 1 - white, 2 - black, 3 - selected]*/
void ChessBoard::addPieceToSquare(ChessPiece *piece, int rank, int file, int color)
{
    // These values are set based on cosmetic appearance
    int offsetX = 5, offsetY = 5, shrinkX = 10, shrinkY = 10;

    // Create and scale the sprite
    QPixmap pieceSprite;
    if (color == 2) { pieceSprite = piece->getDarkIcon(); }
    else if (color == 1) { pieceSprite = piece->getLightIcon(); }
    else if (color == 3) { pieceSprite = piece->getSelectedIcon(); }
    ChessSquare *squares = boardSquares[rank][file];
    QPixmap scaledPiece = pieceSprite.scaled(tileSize-shrinkX, tileSize-shrinkY, Qt::KeepAspectRatio);
    QGraphicsPixmapItem *finalSprite = new QGraphicsPixmapItem(scaledPiece);

    // Position and add to scene
    finalSprite->setPos(squares->rect().topLeft() + QPointF(offsetX, offsetY));
    chessScene->addItem(finalSprite);

    // Add to associated ChessSquare object
    ChessSquare *square = boardSquares[rank][file];
    square->setOccupyingPiece(piece);
    piece->setSprite(finalSprite);

    return;
}

/* Remove sprite from square and set occupying piece to nullptr */
void ChessBoard::removePieceFromSquare(ChessSquare *square)
{
    if (square->getOccupyingPiece() == nullptr) {
        qDebug() << "Attempting to remove nullptr from square -- aborting.";
        return;
    } else {
        ChessPiece *piece = square->getOccupyingPiece();
        QGraphicsPixmapItem *sprite = piece->getSprite();
        if (sprite) {
            qDebug() << "Removing sprite from square.";
            chessScene->removeItem(sprite);
        } else {
            qDebug() << "Sprite pointer is null.";
        }
        square->setOccupyingPiece(nullptr);
    }
    return;
}

/* Return reference to the ChessSquare object at specified rank and file */
ChessSquare* ChessBoard::getSquare(int rank, int file)
{
    return boardSquares[rank][file];
}

/* Alters necessary flag and swaps piece icon */
void ChessBoard::selectPiece(ChessPiece *selectedPiece, ChessSquare *square) {
    selectedPiece->setIsSelected(true);
    removePieceFromSquare(square);
    addPieceToSquare(selectedPiece, square->getRank(), square->getFile(), 3);

    return;
}

/* Alters necessary flag and swaps piece icon */
void ChessBoard::deselectPiece(ChessPiece *selectedPiece, ChessSquare *square) {
    selectedPiece->setIsSelected(false);
    int pieceBaseColor = selectedPiece->getWhite() == true ? 1 : 2;
    removePieceFromSquare(square);
    addPieceToSquare(selectedPiece, square->getRank(), square->getFile(), pieceBaseColor);

    return;
}

/* Runs only if square clicked has piece on it! Highlight all potential moves based on the piece selected */
void ChessBoard::highlightPossibleSquares(ChessSquare *square) {
    ChessPiece *selectedPiece = square->getOccupyingPiece();
    selectPiece(selectedPiece, square);

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
            continue;
        }

        if (lineStopped == false) {
            if (selectedPiece->getWhite()) {
                // Light pieces [Currently opponent side]
                newFile = square->getFile() - x;          // Change in x-axis
                newRank = square->getRank() + y;     // Change in y-axis
            } else {
                // Dark pieces [Currently player side]
                newFile = square->getFile() + x;          // Change in x-axis
                newRank = square->getRank() - y;     // Change in y-axis
            }

            // Only move forward if coordinate is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                bool squareOccupied = possibleMove->getOccupyingPiece() == nullptr ? false : true;
                if (squareOccupied == true) { qDebug()<< "Square occupied by:" << possibleMove->getOccupyingPiece()->getName(); }

                if (squareOccupied == false) {
                    // If square is not occupied, it is a potential move
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                    movePending = true;
                } else if (selectedPiece->getWhite() == true && possibleMove->getOccupyingPiece()->getWhite() == false) {
                    // If selected piece is white and target piece is black, it is potential move but blocks the line
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                    movePending = true;
                    lineStopped = true;
                } else if (selectedPiece->getWhite() == false && possibleMove->getOccupyingPiece()->getWhite() == true) {
                    // If selected piece is black and target piece is white, it is potential move but blocks the line
                    possibleMove->toggleSquareYellow();
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                    movePending = true;
                    lineStopped = true;
                } else {
                    // If square is occupied by friendly, that is blocks the line
                    lineStopped = true;
                }
            }
        }
    }
    return;
}

/* Deselect piece, empty out highlight and move vectors, reset base square color */
void ChessBoard::movePiece(ChessSquare *squareClicked)
{
    ChessPiece *pieceToMove = selectedSquare->getOccupyingPiece(); // Get piece from old square
    removePieceFromSquare(selectedSquare); // Remove sprite from old square
    deselectPiece(pieceToMove, squareClicked); // Add sprite to new square and deselect piece
    chessScene->update();

    // Reset all move variables
    resetPossibleMoveSquares();
    resetHighlightedSquares();
    selectedSquare->resetColor();
    selectedSquare = nullptr;
    movePending = false;
    return;
}

/* This function runs if the user left clicks on a square */
void ChessBoard::squareLeftClicked(int rank, int file)
{
    if (selectedSquare != nullptr) { qDebug() << "Existing selected square on click:" << selectedSquare->getRank() << selectedSquare->getFile() << "."; } else { qDebug() << "No selected square on click."; }

    // Check that click was legal
    if (boardSquares[rank][file] == nullptr) {
        qDebug() << "User has somehow clicked on a square that does not exist";
    } else {
        ChessSquare *squareClicked = this->getSquare(rank, file);

        // If move currently pending
        if (movePending) {
            if (squareInPossibleMoves(squareClicked)) {
                qDebug() << "Move pending and moving piece.";
                movePiece(squareClicked); // Move piece will deselect piece, empty out highlight and move vectors, reset base square color
            } else {
                qDebug() << "Move pending but not moving piece.";
                movePending = false;
                deselectPiece(selectedSquare->getOccupyingPiece(), selectedSquare);
                resetPossibleMoveSquares();
                resetHighlightedSquares();

                // If the square is not already selected
                if (selectedSquare != squareClicked) {
                    selectSquare(squareClicked);
                } else { // Else if the square was already actively selected (deselect square)
                    deselectSquare(squareClicked);
                }
            }
        } else {
            qDebug() << "No move pending";
            // Reset data from previous click
            resetPossibleMoveSquares();
            resetHighlightedSquares();

            if (selectedSquare != nullptr) {
                // If the square is not already selected
                if (selectedSquare != squareClicked) {
                    if (selectedSquare->getOccupyingPiece() != nullptr) {
                        deselectPiece(selectedSquare->getOccupyingPiece(), selectedSquare);
                    }
                    selectSquare(squareClicked);
                    qDebug() << "No move pending and square not already selected.";
                } else { // Else if the square was already actively selected (deselect square)
                    deselectSquare(squareClicked);
                    if (squareClicked->getOccupyingPiece() != nullptr) {
                        deselectPiece(squareClicked->getOccupyingPiece(), squareClicked);
                    }
                    qDebug() << "No move pending and square was already selected.";
                }
            } else {
                selectSquare(squareClicked);
                qDebug() << "No move pending and no square already selected.";
            }
        }
    }


    qDebug() << "----------------------------------------------------------------";

    return;
}

void ChessBoard::selectSquare(ChessSquare *squareClicked) {
    if (selectedSquare != nullptr) {
        selectedSquare->resetColor();
    }
    selectedSquare = squareClicked;
    squareClicked->toggleSquareYellow();
    if (squareClicked->getOccupyingPiece() != nullptr) {
        highlightPossibleSquares(squareClicked);
    }

    return;
}

void ChessBoard::deselectSquare(ChessSquare * squareClicked) {
    if (squareClicked != nullptr) {
        squareClicked->resetColor();
    }
    selectedSquare = nullptr;

    return;
}

/* Returns true if the specified chess quare object is a valid place to move */
bool ChessBoard::squareInPossibleMoves(ChessSquare *square)
{
    for (ChessSquare *s : possibleMoveSquares) {
        if (s == square) {
            return true;
        }
    }
    return false;
}

/* Reset all highlighted squares to original color */
void ChessBoard::resetHighlightedSquares() {
    while (!highlightedSquares.empty()) {
        ChessSquare *square = highlightedSquares.back();
        highlightedSquares.pop_back();
        square->resetColor();
    }
    return;
}

/* Empty the possible moves vector */
void ChessBoard::resetPossibleMoveSquares() {
    while(!possibleMoveSquares.empty()) {
        possibleMoveSquares.pop_back();
    }
    return;
}

void ChessBoard::printMoveDebug(QString header) {
    qDebug() << header;
    if (selectedSquare != nullptr) { qDebug() << "Selected square:" << selectedSquare->getRank() << selectedSquare->getFile(); } else { qDebug() << "No selected square"; }
    if (!highlightedSquares.empty()) { qDebug() << "Highlighted squares is NOT empty."; } else { qDebug() << "Highlighted squares is empty."; }
    if (!possibleMoveSquares.empty()) { qDebug() << "Possible moves is NOT empty"; } else { qDebug() << "Possible moves is empty"; }
    qDebug() << Qt::endl;

    return;
}
