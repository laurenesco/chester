//
// Program Name:              chessboard.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessBoard class. See header file for details.
//

#include <regex>

#include "chessboard.h"
#include "logic.h"
#include "pythoninterface.h"
#include "qboxlayout.h"
#include <cmath>

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

// ---------------------------------------- //
// CHESS BOARD RELATED FUNCTIONS //
// ---------------------------------------- //

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
            connect(square, &ChessSquare::squareRightClicked, this, &ChessBoard::squareRightClicked);
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

            if (rank == 6) {
                pawn->setWhite(true);
            } else {
                pawn->setWhite(false);
            }

            int color = pawn->getWhite() == true ? 1 : 2;
            addPieceToSquare(pawn, rank, j, color);
        }
    }

    // Rooks
    int rookFlag = 0;
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;

            Rook *rook = new Rook();

            if (rank == 7) {
                // Rank 7, white pieces
                rook->setWhite(true);
                if (file == 0) {
                    // File 0, kingside
                    whiteKingsideRook = rook;
                } else {
                    // File 7, queenside
                    whiteQueensideRook = rook;
                }
            } else {
                // Rank 0, black pieces
                rook->setWhite(false);
                if (file == 0) {
                    // File 0, kingside
                    blackKingsideRook = rook;
                } else {
                    // File 7, queenside
                    blackQueensideRook = rook;
                }
            }

            int color = rook->getWhite() == true ? 1 : 2;
            addPieceToSquare(rook, rank, file, color);
            rookFlag ++;
        }
    }

    //  Knights
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;

            Knight *knight = new Knight();

            if (rank == 7) {
                knight->setWhite(true);
            } else {
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

            if (rank == 7) {
                bishop->setWhite(true);
            } else {
                bishop->setWhite(false);
            }

            int color = bishop->getWhite() == true ? 1 : 2;
            addPieceToSquare(bishop, rank, file, color);
        }
    }

    // Kings
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 1; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;

            King *king = new King();

            if (rank == 7) {
                whiteKing = king;
                king->setWhite(true);
            } else {
                blackKing = king;
                king->setWhite(false);
            }

            int color = king->getWhite() == true ? 1 : 2;
            addPieceToSquare(king, rank, file, color);
        }
    }

    // Queens
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 1; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;

            Queen *queen = new Queen();

            if (rank == 7) {
                queen->setWhite(true);
            }
            else {
                queen->setWhite(false);
            }

            int color = queen->getWhite() == true ? 1 : 2;
            addPieceToSquare(queen, rank, file, color);
        }
    }
}

std::vector<ChessSquare*> ChessBoard::getNextMove(QString UCI)
{
    PythonInterface *python = new PythonInterface();
    std::string nextMove = python->getNextMove(UCI).toStdString();
    qDebug() << "Next best move:" << nextMove;

    std::vector<ChessSquare*> moveSquares;

    std::string startFile, endFile, startRank, endRank;

    startFile = nextMove.substr(0, 1);
    startRank = nextMove.substr(1, 1);
    endFile = nextMove.substr(2, 1);
    endRank = nextMove.substr(3, 1);

//    qDebug() << startFile << startRank;
//    qDebug() << endFile << endRank;

    int startRankInt = abs((startRank[0] - 48) - 8);
    int endRankInt = abs((endRank[0] - 48) - 8);
    int startFileInt = startFile[0] - 97;
    int endFileInt = endFile[0] - 97;

    // qDebug() << "Start Rank:" << startRankInt << "Start File:" << startFileInt;
    // qDebug() << "End Rank:" << endRankInt << "End File:" << endFileInt;

    ChessSquare *startSquare = this->getSquare(startRankInt, startFileInt);
    ChessSquare *endSquare = this->getSquare(endRankInt, endFileInt);

    moveSquares.push_back(endSquare);
    moveSquares.push_back(startSquare);

    return moveSquares;
}

// ----------------------- //
// GET / SET METHODS //
// ---------------------- //

/* Return reference to the ChessSquare object at specified rank and file */
ChessSquare* ChessBoard::getSquare(int rank, int file)
{
    return boardSquares[rank][file];
}

// ------------------------------- //
// LOGIC RELATED FUNCTIONS //
// ------------------------------- //

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

/* Reset all yellow squares to original color */
void ChessBoard::resetHighlightedSquares() {
    while (!highlightedSquares.empty()) {
        ChessSquare *square = highlightedSquares.back();
        highlightedSquares.pop_back();
        square->resetColor();
    }
    return;
}

/* Reset all red squares to original color */
void ChessBoard::resetRedSquares() {
    while (!redSquares.empty()) {
        ChessSquare *square = redSquares.back();
        redSquares.pop_back();
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

// ------------------------------------------//
// CHESS SQUARE RELATED FUNCTIONS //
// ------------------------------------------//

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

        // Pull set of coordinates from vector
        int x = coords[i];
        int y = coords[i+1];

        // Reset lineStopped when a new direction is started (marked by 0, 0 coords)
        if (x == 0 && y == 0) {
            lineStopped = false;
            continue;
        }

        if (lineStopped == false) {
            if (selectedPiece->getWhite() != true) {
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
                // if (squareOccupied == true) { qDebug()<< "Square occupied by:" << possibleMove->getOccupyingPiece()->getName(); }

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

/* This slot runs if the user left clicks on a square */
void ChessBoard::squareLeftClicked(int rank, int file)
{
    // Check that click was legal
    if (boardSquares[rank][file] == nullptr) {
        qDebug() << "User has somehow clicked on a square that does not exist";
    } else {
        ChessSquare *squareClicked = this->getSquare(rank, file);

        // If move currently pending
        if (movePending) {
            if (squareInPossibleMoves(squareClicked)) {
                // qDebug() << "Move pending and moving piece.";
                movePiece(squareClicked); // Move piece will deselect piece, empty out highlight and move vectors, reset base square color
            } else {
                // qDebug() << "Move pending but not moving piece.";
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
            // qDebug() << "No move pending";
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
                    // qDebug() << "No move pending and square not already selected.";
                } else { // Else if the square was already actively selected (deselect square)
                    deselectSquare(squareClicked);
                    if (squareClicked->getOccupyingPiece() != nullptr) {
                        deselectPiece(squareClicked->getOccupyingPiece(), squareClicked);
                    }
                    // qDebug() << "No move pending and square was already selected.";
                }
            } else {
                selectSquare(squareClicked);
                // qDebug() << "No move pending and no square already selected.";
            }
        }
    }

    // Move black piece
    if (whiteToPlay == false) {
         qDebug() << "----------------------------------------------------------------";
        qDebug() << "Black moving...";
        QString UCI = this->boardToUCI();

        std::vector<ChessSquare*> moveSquares = this->getNextMove(UCI);
        selectedSquare = moveSquares[1];
        this->movePiece(moveSquares[0]);
        selectedSquare = nullptr;

        whiteToPlay = true;
    }

    qDebug() << "----------------------------------------------------------------";

    return;
}

// This slot runs if the user right clicks on a square, it:
//   - highlights the square red
//   - adds the squars to redSquares vector
void ChessBoard::squareRightClicked(int rank, int file) {
    ChessSquare *square = boardSquares[rank][file];
    QColor color = QColor(164,90,82);
    square->toggleSquareCustom(color);
    redSquares.push_back(square);
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

// -------------------------------------- //
// CHESS PIECE RELATED FUNCTIONS //
// -------------------------------------- //

// This function is the driver funtion for moving pieces, and is responsible to:
// - move the piece sprite
// - deselect piece
// - empty out highlighted square vector
// - empty out possible move vector
// - reset selectedsquare color
// - set selectedsquare to nullptr
// - set movepending to false
// - toggle pawn moved flag
// - toggle rook moved flag
// - toggle king moved flag
// - update half move counter
// - update full move counter
// - save lastmove in algebraic notation
// - update what color is to play
// - update lastmovedpiece
// - check for en passant status
//
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

    if (pieceToMove->getName() == "Pawn") {
        // Pawn specific (en passant)
        Pawn* pawn = dynamic_cast<Pawn*>(pieceToMove);
        pawn->incrementMoveCounter();
    } else if (pieceToMove->getName() == "Rook") {
        // Rook specific (castling)
        Rook *rook = dynamic_cast<Rook*>(pieceToMove);
        if (rook->moved == false) {
            rook->moved = true;
        }
    } else if (pieceToMove->getName() == "King") {
        // King specific (castling)
        King *king = dynamic_cast<King*>(pieceToMove);
        if (king->moved == false) {
            king->moved = true;
        }
    }

    // If en passant capture, remove captured pawn
    if (squareClicked == enPassantSquare) {
        int rank = enPassantPiece->getWhite() == true ? enPassantSquare->getRank() - 1 : enPassantSquare->getRank() + 1;
        removePieceFromSquare(boardSquares[rank][enPassantSquare->getFile()]);
    }

    // Update game counter
    if (pieceToMove->getWhite() == true) {
        halfMoveCounter ++;
    } else {
        fullMoveCounter += 2;
    }

    lastMove = moveToAlgebraic(pieceToMove, squareClicked);
    whiteToPlay = whiteToPlay == true ? false : true;
    lastMovedPiece = pieceToMove;
    manageEnPassant();

    Q_EMIT moveCompleted(lastMove);

    return;
}

// This function is responsible for:
// - updating the pieces "selected" flag
// - changing the color of the piece on the board to the selected color
void ChessBoard::selectPiece(ChessPiece *selectedPiece, ChessSquare *square) {
    selectedPiece->setIsSelected(true);

    // Remove and add piece to change sprite color (can probably be done better)
    removePieceFromSquare(square);
    addPieceToSquare(selectedPiece, square->getRank(), square->getFile(), 3);

    return;
}

/* This function is responsible for:
 * - updating the pieces "selected" flag
 * - changing the color of the piece on the board back to its base color
 */
void ChessBoard::deselectPiece(ChessPiece *selectedPiece, ChessSquare *square) {
    selectedPiece->setIsSelected(false);
    int pieceBaseColor = selectedPiece->getWhite() == true ? 1 : 2;
    removePieceFromSquare(square);
    addPieceToSquare(selectedPiece, square->getRank(), square->getFile(), pieceBaseColor);

    return;
}

/* This function is responsible for:
 * - creating a sprite of the piece in the appropriate size and color
 *   [Color code: 1 - white, 2 - black, 3 - selected]
 * - placing it centered on the correct square
 * - update the occupyingpiece pointer of said square
 * - update the sprite pointer for the piece
 */
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

// This function is responsible for:
// - remove the sprite from the square
// - set the occupyingpiece pointer of square to nullptr
void ChessBoard::removePieceFromSquare(ChessSquare *square)
{
    ChessPiece *piece = square->getOccupyingPiece();
    if (piece == nullptr) {
        // qDebug() << "Attempting to remove nullptr from square -- aborting.";
        return;
    } else {
        QGraphicsPixmapItem *sprite = piece->getSprite();
        if (sprite) {
            // qDebug() << "Removing sprite from square.";
            chessScene->removeItem(sprite);
        } else {
            // qDebug() << "Sprite pointer is null.";
        }
        square->setOccupyingPiece(nullptr);
    }
    return;
}

// ----------------------------------------- //
// CHESS STATUS RELATED FUNCTIONS //
// ---------------------------------------- //

// This function checks en passant status by:
// -
//
// Restrictions: only detects one en passant square at a time
// Note: this function utilizes normal chess rank and file numberings
void ChessBoard::manageEnPassant()
{
    QColor color = QColor(241,235,156);
    if (enPassantSquare != nullptr) { enPassantCounter ++; }

    // Reset en passant status if expired
    if (enPassantCounter > 1) {
        enPassantCounter = 0;
        enPassantPiece = nullptr;
        enPassantSquare->resetColor();
        enPassantSquare = nullptr;
    }

    // If a pawn moved last
    if (lastMovedPiece->getName() == "Pawn") {
        Pawn* pawn = dynamic_cast<Pawn*>(lastMovedPiece);
        std::string lastMoveString = lastMove.toStdString();
        qDebug() << "Last move string:" << lastMoveString;

        // If it was the pawns first move, get the rank the pawn is on from the last move (algebraic notation)
        if (pawn->getMoveCounter() == 1) {
            std::string rankString = lastMoveString.substr(lastMoveString.size()-1, 1);
            int vulnerableRank;
            std::stringstream(rankString) >> vulnerableRank;
            qDebug() << "Vulnerable rank:" << vulnerableRank;

            // If it was a two-space move
            if (vulnerableRank == 4 || vulnerableRank == 5) {
                if(vulnerableRank == 5) { vulnerableRank = 3; }

                // Determine the possible attacking files (the files to the left and right of the pawn)
                int rightAttackFile = static_cast<int>(lastMoveString.substr(lastMoveString.size()-2, 1)[0]) - 96;
                int leftAttackFile = static_cast<int>(lastMoveString.substr(lastMoveString.size()-2, 1)[0]) - 98;
                qDebug() << "Left attack file:" << leftAttackFile << " Right attack file:" << rightAttackFile;

                // If the left attack file is on the board and occupied by an opposing pawn
                if (leftAttackFile >= 0) {
                    ChessPiece *attackingPiece = boardSquares[vulnerableRank][leftAttackFile]->getOccupyingPiece();
                    if (attackingPiece != nullptr && attackingPiece->getName() == "Pawn" && attackingPiece->getWhite() != pawn->getWhite()) {
                        // Target rank depends on piece color
                        if (pawn->getWhite() == true) {
                            vulnerableRank = vulnerableRank + 1;
                        } else {
                            vulnerableRank = vulnerableRank - 1;
                        }
                        enPassantSquare = boardSquares[vulnerableRank][leftAttackFile + 1];
                        enPassantSquare->toggleSquareCustom(color);
                        enPassantPiece = pawn;
                        qDebug() << "en passant square:" << enPassantSquare->getRank() << enPassantSquare->getFile();
                        return;
                    }
                }
                // If the right attack file is on the board and occupied by an opposing pawn
                if (rightAttackFile <= 7) {
                    ChessPiece *attackingPiece = boardSquares[vulnerableRank][rightAttackFile]->getOccupyingPiece();
                    if (attackingPiece != nullptr && attackingPiece->getName() == "Pawn" && attackingPiece->getWhite() != pawn->getWhite()) {
                        // Target rank depends on piece color
                        if (pawn->getWhite() == true) {
                            vulnerableRank = vulnerableRank + 1;
                        } else {
                            vulnerableRank = vulnerableRank - 1;
                        }
                        enPassantSquare = boardSquares[vulnerableRank][rightAttackFile - 1];
                        enPassantSquare->toggleSquareCustom(color);
                        enPassantPiece = pawn;
                        qDebug() << "en passant square:" << enPassantSquare->getRank() << enPassantSquare->getFile();
                        return;
                    }
                }
            }
        }
    }
    return;
}

QString ChessBoard::getCastlingRights()
{
    // Can update this function with global BQCastle, BKCastle, WKCastle, WQCastle variables to skip iterating through each time
    // once castling is lost, since it is never regained.
    QString uci;
    bool castling = false;

    if (whiteKing->moved == false) {
        if (whiteKingsideRook != nullptr && whiteKingsideRook->moved == false) {
            uci = uci + "K";
            castling = true;
            if (boardSquares[7][1]->getOccupyingPiece() == nullptr && boardSquares[7][2]->getOccupyingPiece() == nullptr) {
                // Logic for adding to possible squares
            }
        }
        if (whiteQueensideRook != nullptr && whiteQueensideRook->moved == false) {
            uci = uci + "Q";
            castling = true;
            if (boardSquares[7][4]->getOccupyingPiece() == nullptr && boardSquares[7][5]->getOccupyingPiece() == nullptr && boardSquares[7][6]->getOccupyingPiece() == nullptr) {
                // Logic for adding to possible squares
            }
        }
    }
    if (blackKing -> moved == false) {
        if (blackKingsideRook != nullptr && blackKingsideRook->moved == false) {
            uci = uci + "k";
            castling = true;
            if (boardSquares[0][1]->getOccupyingPiece() == nullptr && boardSquares[0][2]->getOccupyingPiece() == nullptr) {
                // Logic for adding to possible squares
            }
        }
        if (blackQueensideRook != nullptr && blackQueensideRook->moved == false) {
            uci = uci + "q";
            castling = true;
            if (boardSquares[0][4]->getOccupyingPiece() == nullptr && boardSquares[0][5]->getOccupyingPiece() == nullptr && boardSquares[0][6]->getOccupyingPiece() == nullptr) {
                // Logic for adding to possible squares
            }
        }
    }

    if (castling == false) {
        uci = uci + "-";
    }

    return uci;
}

// ------------------------------------- //
// NOTATION RELATED FUNCTIONS //
// ------------------------------------- //

QString ChessBoard::boardToUCI()
{
    QString uci;

    for (int i = 0; i < 8; i++) {
        // New rank
        int emptySquares = 0;

        for (int j = 0; j < 8; j++) {
            // For each file on that rank
            // If that square has a piece
            if (boardSquares[i][j]->getOccupyingPiece() != nullptr) {
                // Decide whether to precede with ".", number, or nothing
                if (emptySquares >= 1) {
                    uci = uci + QString::fromStdString(std::to_string(emptySquares));
                }
                uci = uci + boardSquares[i][j]->getOccupyingPiece()->getFEN();
                emptySquares = 0;
            } else {
                emptySquares ++;
            }
        }
        // Need to account for trailing empty squares
        if (emptySquares >= 1) {
            uci = uci + QString::fromStdString(std::to_string(emptySquares));
        }
        // End rank
        if (i < 7) {
            uci = uci + "/";
        }
    }
    // Denote whose turn it is
    uci = uci + " ";
    uci = whiteToPlay ? uci + "w" : uci + "b";
    // Denote castling rights
    uci = uci + " ";
    uci = uci + this->getCastlingRights();
    // Denote en passant status
    uci = uci + " ";
    if (enPassantSquare != nullptr) {
        ChessPiece *pawn = new ChessPiece();
        uci = uci + this->moveToAlgebraic(pawn, enPassantSquare);
    } else {
        uci = uci + "-";
    }
    // Denote halfmove and fullmove
   uci = uci + " ";
   uci = uci + QString::fromStdString(std::to_string(halfMoveCounter));
   uci = uci + " ";
   uci = uci + QString::fromStdString(std::to_string(fullMoveCounter));

   qDebug() << uci;

   return uci;
}

/* Convert the most recent move into algebraic notation */
QString ChessBoard::moveToAlgebraic(ChessPiece *piece, ChessSquare *square)
{
    QString p;
    QString name = piece->getName();

    if (name == "Pawn") {
        // pawns dont get letter appended
    } else if (name == "Rook") {
        p = p + "R";
    } else if (name == "Knight") {
        p = p + "N";
    } else if (name == "Bishop") {
        p = p + "B";
    } else if (name == "Queen") {
        p = p + "Q";
    } else if (name == "King") {
        p = p + "K";
    } else {
        qDebug() << "Error getting piece type in moveToAlgebraic()";
    }

    char file = 96 + square->getFile() + 1;
    int rank = 8 - square->getRank();

    QString algebraic = p + file + QString::fromStdString(std::to_string(rank));

    return algebraic;
}
