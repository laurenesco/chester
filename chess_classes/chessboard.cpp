//
// Program Name:              chessboard.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessBoard class. See header file for details.
//

#include <regex>

#include "chessboard.h"
#include "pythoninterface.h"
#include "qboxlayout.h"
#include <cmath>
#include <windows.h>
#include <random>
#include <QElapsedTimer>

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
    this->config = new Config();
    config->refreshConfig();

    createChessBoard();
    loadStartingPosition();
}

void ChessBoard::start_time()
{
    start = clock();
    return;
}

void ChessBoard::end_time(QString name)
{
    end = clock();
    qDebug() << name << "ran for" << double(end - start) / CLOCKS_PER_SEC;
    return;
}

// ---------------------------------------- //
// CHESS BOARD RELATED FUNCTIONS //
// ---------------------------------------- //

// The 64 chess board ChessSquare objects created here
void ChessBoard::createChessBoard() {
    //    qDebug() << "====== Entering createChessBoard";
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            ChessSquare *square = new ChessSquare(file * tileSize, rank * tileSize, tileSize, tileSize, this->config->getColor());
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

// Creates all the ChessPiece objects
void ChessBoard::loadStartingPosition() {
    // Pawns
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;

            Pawn *pawn = new Pawn(this->config->getColor());

            if (rank == 6) {
                if (config->getColor() == true) {
                    pawn->setWhite(true);
                } else {
                    pawn->setWhite(false);
                }
            } else {
                if (config->getColor() == true) {
                    pawn->setWhite(false);
                } else {
                    pawn->setWhite(true);
                }
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

            Rook *rook = new Rook(this->config->getColor());

            if (rank == 7) {
                // Rank 7, white pieces
                if (config->getColor() == true) {
                    rook->setWhite(true);
                } else {
                    rook->setWhite(false);
                }
                if (file == 0) {
                    // File 0, kingside
                    whiteKingsideRook = rook;
                } else {
                    // File 7, queenside
                    whiteQueensideRook = rook;
                }
            } else {
                // Rank 0, black pieces
                if (config->getColor() == true) {
                    rook->setWhite(false);
                } else {
                    rook->setWhite(true);
                }
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

            Knight *knight = new Knight(this->config->getColor());

            if (rank == 7) {
                if (config->getColor() == true) {
                    knight->setWhite(true);
                } else {
                    knight->setWhite(false);
                }
            } else {
                if (config->getColor() == true) {
                    knight->setWhite(false);
                } else {
                    knight->setWhite(true);
                }
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

            Bishop *bishop = new Bishop(this->config->getColor());

            if (rank == 7) {
                if (config->getColor() == true) {
                    bishop->setWhite(true);
                } else {
                    bishop->setWhite(false);
                }
            } else {
                if (config->getColor() == true) {
                    bishop->setWhite(false);
                } else {
                    bishop->setWhite(true);
                }
            }

            int color = bishop->getWhite() == true ? 1 : 2;
            addPieceToSquare(bishop, rank, file, color);
        }
    }

    // Kings
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 1; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;

            King *king = new King(this->config->getColor());

            if (rank == 7) {
                if (config->getColor() == true) {
                    whiteKing = king;
                    king->setWhite(true);
                } else {
                    blackKing = king;
                    king->setWhite(false);
                }
            } else {
                if (config->getColor() == true) {
                    blackKing = king;
                    king->setWhite(false);
                } else {
                    whiteKing = king;
                    king->setWhite(true);
                }
            }

            king->rank = rank;
            king->file = file;

            int color = king->getWhite() == true ? 1 : 2;
            addPieceToSquare(king, rank, file, color);
        }
    }

    // Queens
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 1; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;

            Queen *queen = new Queen(this->config->getColor());

            if (rank == 7) {
                if (config->getColor() == true) {
                    queen->setWhite(true);
                } else {
                    queen->setWhite(false);
                }
            } else {
                if (config->getColor() == true) {
                    queen->setWhite(false);
                } else {
                    queen->setWhite(true);
                }
            }

            int color = queen->getWhite() == true ? 1 : 2;
            addPieceToSquare(queen, rank, file, color);
        }
    }
}

// Connect  to python engine and get the next best move
void ChessBoard::getNextMove(std::string nextMove)
{
    nextBestMoveSquares.clear();

    std::string startFile, endFile, startRank, endRank;

    startFile = nextMove.substr(0, 1);
    startRank = nextMove.substr(1, 1);
    endFile = nextMove.substr(2, 1);
    endRank = nextMove.substr(3, 1);

    int startRankInt = abs((startRank[0] - 48) - 8);
    int endRankInt = abs((endRank[0] - 48) - 8);
    int startFileInt = startFile[0] - 97;
    int endFileInt = endFile[0] - 97;

    ChessSquare *startSquare = this->getSquare(startRankInt, startFileInt);
    ChessSquare *endSquare = this->getSquare(endRankInt, endFileInt);

    nextBestMoveSquares.push_back(endSquare);
    nextBestMoveSquares.push_back(startSquare);

    return;
}

// Connect to python engine and get current evaluation
void ChessBoard::getEvaluation(std::string game_eval) {
    // Process the evaluation
    int separator = game_eval.find('?');

    // Get the game state
    eval.status = (game_eval.substr(separator + 1) == "cp") ? 1 : 2;

    // Get the evaluation value
    std::stringstream(game_eval.substr(0, separator)) >> eval.value;

    // Determine who is winning
    eval.winning = eval.value >= 0 ? 1 : 2;

    // After determining who is winning, now can take abs value of value
    eval.value = std::abs(eval.value);
    return;
}

void ChessBoard::getStats()
{
    PythonInterface *python = new PythonInterface();
    QString UCI = this->boardToUCI();
    std::string result = python->getStats(UCI).toStdString();

    // Separate the results
    int separator = result.find('&');
    std::string nextMove = result.substr(0, separator);
    getNextMove(nextMove);

    std::string game_eval = result.substr(separator+1);
    getEvaluation(game_eval);
    qDebug() << "MOVE:" << nextMove;
    qDebug() << "GAME EVAL:" << game_eval;

    return;
}

std::vector<ChessSquare *> ChessBoard::getPossibleMoves(ChessSquare *square)
{
    ChessPiece *selectedPiece = square->getOccupyingPiece();

    std::vector<ChessSquare*> moves;
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
                // Close side of board
                newFile = square->getFile() - x;          // Change in x-axis
                newRank = square->getRank() + y;     // Change in y-axis
            } else {
                // Far side of board
                newFile = square->getFile() + x;          // Change in x-axis
                newRank = square->getRank() - y;     // Change in y-axis
            }

            // Only move forward if coordinate is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                bool possibleMoveOccupied = possibleMove->getOccupyingPiece() == nullptr ? false : true;

                if (possibleMoveOccupied == false) {
                    // If square is not occupied, it is a potential move
                    moves.push_back(possibleMove);
                } else if (selectedPiece->getWhite() == true && possibleMove->getOccupyingPiece()->getWhite() == false) {
                    // If selected piece is white and target piece is black, it is potential move but blocks the line
                    moves.push_back(possibleMove);
                    lineStopped = true;
                } else if (selectedPiece->getWhite() == false && possibleMove->getOccupyingPiece()->getWhite() == true) {
                    // If selected piece is black and target piece is white, it is potential move but blocks the line
                    possibleMoveSquares.push_back(possibleMove);
                    lineStopped = true;
                } else {
                    // If square is occupied by friendly, that is blocks the line
                    lineStopped = true;
                }
            }
        }
    }

    if (selectedPiece->getName() == "Pawn") {
        // Check attack squares for pawns
        std::vector<int> attackCoords = { -1, 1, 1, 1 };

        // Highlight potential moves yellow
        for (int i = 0; i < (int) attackCoords.size(); i+=2)
        {
            int newRank, newFile;

            // Pull set of coordinates from vector
            int x = coords[i];
            int y = coords[i+1];

            if (selectedPiece->getWhite() != true) {
                // Close side of board
                newFile = square->getFile() - x;          // Change in x-axis
                newRank = square->getRank() + y;     // Change in y-axis
            } else {
                // Far side of board
                newFile = square->getFile() + x;          // Change in x-axis
                newRank = square->getRank() - y;     // Change in y-axis
            }

            // Only move forward if coordinate is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                bool possibleMoveOccupied = possibleMove->getOccupyingPiece() == nullptr ? false : true;
                if (possibleMoveOccupied == true) {
                    possibleMoveSquares.push_back(possibleMove);
                }
            }
        }
    }

    return moves;
}

// TODO place this somewhere and pass the kings rank, file, color, if its coming from check already to it
bool ChessBoard::checkCheck(int kingRank, int kingFile, bool isWhite, bool check) {

    std:: vector<int> directionsVector = {
        1, -1,
        2, -2,
        3, -3,
        4, -4,
        5, -5,
        6, -6,
        7, -7,
        0, 0,
        1, 1,
        2, 2,
        3, 3,
        4, 4,
        5, 5,
        6, 6,
        7, 7,
        0, 0,
        -1, 1,
        -2, 2,
        -3, 3,
        -4, 4,
        -5, 5,
        -6, 6,
        -7, 7,
        0, 0,
        -1, -1,
        -2, -2,
        -3, -3,
        -4, -4,
        -5, -5,
        -6, -6,
        -7, -7,
        0, 0,
        0, -1,
        0, -2,
        0, -3,
        0, -4,
        0, -5,
        0, -6,
        0, -7,
        0, 0,
        0, 1,
        0, 2,
        0, 3,
        0, 4,
        0, 5,
        0, 6,
        0, 7,
        0, 0,
        1, 0,
        2, 0,
        3, 0,
        4, 0,
        5, 0,
        6, 0,
        7, 0,
        0, 0,
        -1, 0,
        -2, 0,
        -3, 0,
        -4, 0,
        -5, 0,
        -6, 0,
        -7, 0,
    };

    bool lineStopped = false;
    int newRank, newFile;

    ChessSquare *kingSquare = boardSquares[kingRank][kingFile];

    // For each coordinate
    for (int i = 0; i < (int) directionsVector.size(); i+=2) {

        // Pull set of coordinates from vector
        int x = directionsVector[i];
        int y = directionsVector[i+1];

        // Reset lineStopped when a new direction is started (marked by 0, 0 coords)
        if (x == 0 && y == 0) {
            lineStopped = false;
            continue;
        }

        // Determine the next square to check
        if (lineStopped == false) {
            if (isWhite != true) {
                // Light pieces
                newFile = kingSquare->getFile() - x;          // Change in x-axis
                newRank = kingSquare->getRank() + y;     // Change in y-axis
            } else {
                // Dark pieces
                newFile = kingSquare->getFile() + x;          // Change in x-axis
                newRank = kingSquare->getRank() - y;     // Change in y-axis
            }

            // Only move forward if coordinate is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *squareToCheck = this->getSquare(newRank, newFile);
                bool squareOccupied = squareToCheck->getOccupyingPiece() == nullptr ? false : true;

                // Only move forward if the square is occupied
                if (squareOccupied == false) {
                    // Do nothing
                } else {
                    ChessPiece *piece = squareToCheck->getOccupyingPiece();

                    if (piece->getWhite() == isWhite) {
                        // If square is occupied by friendly, that is blocks the line
                        lineStopped = true;
                    } else {
                        std::vector<ChessSquare*> moves = getPossibleMoves(squareToCheck);

                        // If the kings square is in the pieces possible moves, then check is present
                        for (ChessSquare* testingSquare : moves) {
                            if (kingSquare == testingSquare) {
                                // CHECK
                                // Check for any escape
                                if (check == false) {
                                    QColor color = QColor(188, 143, 143);
                                    if (config->getAssistModeOn() == true) { kingSquare->toggleSquareCustom(color); }
                                    bool endgame = checkmateCheck(kingSquare, isWhite);
                                    if (endgame == true) {
                                        endGame(kingSquare);
                                    }
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool ChessBoard::checkmateCheck(ChessSquare *kingSquare, bool isWhite)
{
    // Get the possible moves that the king has
    std::vector<ChessSquare*> kingsMoves = getPossibleMoves(kingSquare);
    bool danger = true;

    // If no moves then mate
    if (kingsMoves.empty() == true) {
        endGame(kingSquare);
    } else {
        // Check to see if any possible squares are safe
        for (ChessSquare* testingSquare : kingsMoves) {
            bool danger = checkCheck(testingSquare->getRank(), testingSquare->getFile(), isWhite, true);
            if (danger == false) {
                QColor color = QColor(176,196,222);
                if (config->getAssistModeOn() == true) { testingSquare->toggleSquareCustom(color); }
                return false;
            }
        }
    }
    return danger;
}

void ChessBoard::endGame(ChessSquare *square)
{
    ChessPiece *loser = square->getOccupyingPiece();
    QString color = loser->getWhite() == true ? "White" : "Black";
    QString notification = "Game Over! " + color + " loses!";

    // Disable all chess squares
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            ChessSquare *square = boardSquares[rank][file];
            square->setDisabled(true);
        }
    }

    Q_EMIT game_over(notification);

    return;
}

// ----------------------- //
// GET / SET METHODS //
// ---------------------- //

// Return reference to the ChessSquare object at specified rank and file
ChessSquare* ChessBoard::getSquare(int rank, int file)
{
    return boardSquares[rank][file];
}

// ------------------------------- //
// LOGIC RELATED FUNCTIONS //
// ------------------------------- //

// Returns true if the specified chess quare object is a valid place to move
bool ChessBoard::squareInPossibleMoves(ChessSquare *square)
{
    for (ChessSquare *s : possibleMoveSquares) {
        if (s == square) {
            return true;
        }
    }
    return false;
}

// Reset all yellow squares to original color
void ChessBoard::resetHighlightedSquares() {
    while (!highlightedSquares.empty()) {
        ChessSquare *square = highlightedSquares.back();
        highlightedSquares.pop_back();
        square->resetColor();
    }
    return;
}

// Reset all red squares to original color
void ChessBoard::resetRedSquares() {
    while (!redSquares.empty()) {
        ChessSquare *square = redSquares.back();
        redSquares.pop_back();
        square->resetColor();
    }
    return;
}

// Empty the possible moves vector
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

// Runs only if square clicked has piece on it! Highlight all potential moves based on the piece selected
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
                if (config->getColor() == true) {
                    // Opp side
                    newFile = square->getFile() - x;          // Change in x-axis
                    newRank = square->getRank() + y;     // Change in y-axis
                } else {
                    newFile = square->getFile() + x;          // Change in x-axis
                    newRank = square->getRank() - y;     // Change in y-axis
                }
            } else {
                // Player side
                if (config->getColor() == true) {
                    newFile = square->getFile() + x;          // Change in x-axis
                    newRank = square->getRank() - y;     // Change in y-axis
                } else {
                    newFile = square->getFile() - x;          // Change in x-axis
                    newRank = square->getRank() + y;     // Change in y-axis
                }
            }

            // Only move forward if coordinate is on the board
            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
                ChessSquare *possibleMove = this->getSquare(newRank, newFile);
                bool squareOccupied = possibleMove->getOccupyingPiece() == nullptr ? false : true;
                // if (squareOccupied == true) { qDebug()<< "Square occupied by:" << possibleMove->getOccupyingPiece()->getName(); }

                if (squareOccupied == false) {
                    // If square is not occupied, it is a potential move
                    if (config->getAssistModeOn() == true) { possibleMove->toggleSquareYellow(); }
                    highlightedSquares.push_back(possibleMove);
                    possibleMoveSquares.push_back(possibleMove);
                    movePending = true;
                } else if (selectedPiece->getWhite() == true && possibleMove->getOccupyingPiece()->getWhite() == false) {
                    // If selected piece is white and target piece is black, it is potential move but blocks the line
                    if (selectedPiece->getName() != "Pawn") {
                        if (config->getAssistModeOn() == true) { possibleMove->toggleSquareYellow(); }
                        highlightedSquares.push_back(possibleMove);
                        possibleMoveSquares.push_back(possibleMove);
                        movePending = true;
                        lineStopped = true;
                    }
                } else if (selectedPiece->getWhite() == false && possibleMove->getOccupyingPiece()->getWhite() == true) {
                    // If selected piece is black and target piece is white, it is potential move but blocks the line
                    if (selectedPiece->getName() != "Pawn") {
                        if (config->getAssistModeOn() == true) { possibleMove->toggleSquareYellow(); }
                        highlightedSquares.push_back(possibleMove);
                        possibleMoveSquares.push_back(possibleMove);
                        movePending = true;
                        lineStopped = true;
                    }
                } else {
                    // If square is occupied by friendly, that is blocks the line
                    lineStopped = true;
                }
            }
        }
    }

    if (selectedPiece->getName() == "Pawn") {
        // Check attack squares for pawns
        std::vector<int> attackCoords = { -1, 1, 1, 1 };

        // Highlight potential moves yellow
        for (int i = 0; i < (int) attackCoords.size(); i+=2)
        {
            int attackRank, attackFile;

            // Pull set of coordinates from vector
            int attackx = attackCoords[i];
            int attacky = attackCoords[i+1];

            if (selectedPiece->getWhite() != true) {
                if (config->getColor() == true) {
                    // Opp side
                    attackRank = square->getFile() - attackx;          // Change in x-axis
                    attackFile = square->getRank() + attacky;     // Change in y-axis
                } else {
                    attackFile = square->getFile() + attackx;          // Change in x-axis
                    attackRank = square->getRank() - attacky;     // Change in y-axis
                }
            } else {
                // Player side
                if (config->getColor() == true) {
                    attackFile = square->getFile() + attackx;          // Change in x-axis
                    attackRank = square->getRank() - attacky;     // Change in y-axis
                } else {
                    attackFile = square->getFile() - attackx;          // Change in x-axis
                    attackRank = square->getRank() + attacky;     // Change in y-axis
                }
            }

            // Only move forward if coordinate is on the board
            if (attackRank < 8 && attackFile < 8 && attackRank >= 0 && attackFile >= 0) {
                ChessSquare *attackMove = this->getSquare(attackRank, attackFile);
                bool possibleMoveOccupied = attackMove->getOccupyingPiece() == nullptr ? false : true;
                if (possibleMoveOccupied == true) {
                    if (config->getAssistModeOn() == true) { attackMove->toggleSquareYellow(); }
                    possibleMoveSquares.push_back(attackMove);
                    highlightedSquares.push_back(attackMove);
                }
            }
        }
    }
    return;
}

// This slot runs if the user left clicks on a square
void ChessBoard::squareLeftClicked(int rank, int file)
{
    checkCheck(whiteKing->rank, whiteKing->file, true, false);
    checkCheck(blackKing->rank, blackKing->file, false, false);

    // Check that click was legal
    if (boardSquares[rank][file] == nullptr) {
        qDebug() << "User has somehow clicked on a square that does not exist";
    } else {
        ChessSquare *squareClicked = this->getSquare(rank, file);

        // If move currently pending
        if (movePending == true) {
            if (squareInPossibleMoves(squareClicked) == true) {
                movePiece(squareClicked); // Move piece will deselect piece, empty out highlight and move vectors, reset base square color
                // getStats(); // TODO
                Q_EMIT moveCompleted(lastMove, eval.winning, eval.value);
                // moveBlack();
                checkCheck(whiteKing->rank, whiteKing->file, true, false);
                checkCheck(blackKing->rank, blackKing->file, false, false);
            } else {
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
                } else { // Else if the square was already actively selected (deselect square)
                    deselectSquare(squareClicked);
                    if (squareClicked->getOccupyingPiece() != nullptr) {
                        deselectPiece(squareClicked->getOccupyingPiece(), squareClicked);
                    }
                }
            } else {
                selectSquare(squareClicked);
            }
        }
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
        ChessPiece *piece = squareClicked->getOccupyingPiece();
        bool white = piece->getWhite();
        if (white == true && whiteToPlay == true) {
            highlightPossibleSquares(squareClicked);
        } else if (white == true && whiteToPlay == false) {

        } else if (white == false && whiteToPlay == false) {
            highlightPossibleSquares(squareClicked);
        } else {

        }
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
    lastMove = moveToAlgebraic(pieceToMove, squareClicked);

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
        // King specific (castling/check)
        King *king = dynamic_cast<King*>(pieceToMove);
        if (king->moved == false) {
            king->moved = true;
        }
        king->rank = squareClicked->getRank();
        king->file = squareClicked->getFile();
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

    whiteToPlay = whiteToPlay == true ? false : true;
    lastMovedPiece = pieceToMove;
    manageEnPassant();

    // getStats();

    return;
}

void ChessBoard::moveBlack()
{
    qDebug() << "----------------------------------------------------------------";
    qDebug() << "Black moving...";

    //    Q_EMIT switchMascot(1);
    // Randomly sleep to simulate thinking of computer
    QElapsedTimer timer;
    timer.start();

    std::random_device random_gen;
    std::mt19937 gen(random_gen());
    std::uniform_int_distribution<int> distribution(2, 5);
    int random_number = distribution(gen);
    while (timer.elapsed() < random_number * 1000) {

    }

    selectedSquare = nextBestMoveSquares[1];
    this->movePiece(nextBestMoveSquares[0]);
    selectedSquare = nullptr;

    whiteToPlay = true;

    // getStats(); // TODO
    //    Q_EMIT switchMascot(0);
    Q_EMIT moveCompleted(lastMove, eval.winning, eval.value);

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
 * - literally moving the piece
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
                        if (config->getAssistModeOn() == true) { enPassantSquare->toggleSquareCustom(color); }
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
                        if (config->getAssistModeOn() == true) { enPassantSquare->toggleSquareCustom(color); }
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
        ChessPiece *pawn = new ChessPiece(this->config->getColor());
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
QString ChessBoard::moveToAlgebraic(ChessPiece *movedPiece, ChessSquare *square)
{
    QString piece, algebraic;
    QString name = movedPiece->getName();

    if (name == "Pawn") {
        // pawns dont get letter appended
    } else if (name == "Rook") {
        piece = piece + "R";
    } else if (name == "Knight") {
        piece = piece + "N";
    } else if (name == "Bishop") {
        piece = piece + "B";
    } else if (name == "Queen") {
        piece = piece + "Q";
    } else if (name == "King") {
        piece = piece + "K";
    } else {
        qDebug() << "Error getting piece type in moveToAlgebraic()";
    }

    char file = 96 + square->getFile() + 1;
    int rank = 8 - square->getRank();

    if (square->getOccupyingPiece() != nullptr) {
        algebraic = piece + "x" + file + QString::fromStdString(std::to_string(rank));
    } else {
        algebraic = piece + file + QString::fromStdString(std::to_string(rank));
    }

    return algebraic;
}
