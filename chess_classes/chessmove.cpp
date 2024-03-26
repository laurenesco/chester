//#include "chessmove.h"
//#include "piece_classes//pawn.h"
//#include <QGraphicsScene>

//ChessMove::ChessMove(){
//}

//// This function is reponsible for ensuring that...
////      - the board is valid (two kings on board?)
////      - the move was legal for that piece
////      - any new checks are notated
////      - any captures are completed
////      - en passant status is notated
////      - castling rights are updated
////      - any consecutive repeat moves are notated
////      - stalemates are caught

///* This function runs if the user left clicks on a square */
//bool ChessMove::initiateMove(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare* (&boardSquares)[8][8], bool o_movePending,
//                               std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares, ChessSquare *selectedSquare)
//{
//    // if (selectedSquare != nullptr) { qDebug() << "Existing selected square on click:" << selectedSquare->getRank() << selectedSquare->getFile() << "."; } else { qDebug() << "No selected square on click."; }

//    movePending = o_movePending;

//    // Check that click was legal
//    if (squareClicked == nullptr) {
//        qDebug() << "User has somehow clicked on a square that does not exist";
//    } else {

//        // If move currently pending
//        if (movePending) {
//            if (squareInPossibleMoves(squareClicked, possibleMoveSquares)) {
//                qDebug() << "Move pending and moving piece.";
//                movePiece(squareClicked, chessScene, selectedSquare, movePending); // Move piece will deselect piece, reset base square color
//                resetPossibleMoveSquares(possibleMoveSquares);
//                resetHighlightedSquares(highlightedSquares);
//            } else {
//                qDebug() << "Move pending but not moving piece.";
//                movePending = false;
//                deselectPiece(selectedSquare->getOccupyingPiece(), selectedSquare, chessScene);
//                resetPossibleMoveSquares(possibleMoveSquares);
//                resetHighlightedSquares(highlightedSquares);

//                // If the square is not already selected
//                if (selectedSquare != squareClicked) {
//                    selectSquare(squareClicked, chessScene, selectedSquare, boardSquares, movePending, possibleMoveSquares, highlightedSquares);
//                } else { // Else if the square was already actively selected (deselect square)
//                    deselectSquare(squareClicked, selectedSquare);
//                }
//            }
//        } else {
//            qDebug() << "No move pending";
//            // Reset data from previous click
//            resetPossibleMoveSquares(possibleMoveSquares);
//            resetHighlightedSquares(highlightedSquares);

//            if (selectedSquare != nullptr) {
//                // If the square is not already selected
//                if (selectedSquare != squareClicked) {
//                    if (selectedSquare->getOccupyingPiece() != nullptr) {
//                        deselectPiece(selectedSquare->getOccupyingPiece(), selectedSquare, chessScene);
//                    }
//                    selectSquare(squareClicked, chessScene, selectedSquare, boardSquares, movePending, possibleMoveSquares, highlightedSquares);
//                    qDebug() << "No move pending and square not already selected.";
//                } else { // Else if the square was already actively selected (deselect square)
//                    deselectSquare(squareClicked, selectedSquare);
//                    if (squareClicked->getOccupyingPiece() != nullptr) {
//                        deselectPiece(squareClicked->getOccupyingPiece(), squareClicked, chessScene);
//                    }
//                    qDebug() << "No move pending and square was already selected.";
//                }
//            } else {
//                selectSquare(squareClicked, chessScene, selectedSquare, boardSquares, movePending, possibleMoveSquares, highlightedSquares);
//                qDebug() << "No move pending and no square already selected.";
//            }
//        }
//    }
//    return movePending;
//}

///* Add a given chess piece to a square specified by rank and file [Color code: 1 - white, 2 - black, 3 - selected]*/
//void ChessMove::addPieceToSquare(ChessPiece *piece, ChessSquare *square, int color, QGraphicsScene *chessScene)
//{
//    // These values are set based on cosmetic appearance
//    int offsetX = 5, offsetY = 5, shrinkX = 10, shrinkY = 10;

//    // Create and scale the sprite
//    QPixmap pieceSprite;
//    if (color == 2) { pieceSprite = piece->getDarkIcon(); }
//    else if (color == 1) { pieceSprite = piece->getLightIcon(); }
//    else if (color == 3) { pieceSprite = piece->getSelectedIcon(); }
//    QPixmap scaledPiece = pieceSprite.scaled(tileSize-shrinkX, tileSize-shrinkY, Qt::KeepAspectRatio);
//    QGraphicsPixmapItem *finalSprite = new QGraphicsPixmapItem(scaledPiece);

//    // Position and add to scene
//    finalSprite->setPos(square->rect().topLeft() + QPointF(offsetX, offsetY));
//    chessScene->addItem(finalSprite);

//    // Add to associated ChessSquare object
//    square->setOccupyingPiece(piece);
//    piece->setSprite(finalSprite);

//    return;
//}

///* Remove sprite from square and set occupying piece to nullptr */
//void ChessMove::removePieceFromSquare(ChessSquare *square , QGraphicsScene *chessScene)
//{
//    if (square->getOccupyingPiece() == nullptr) {
//        qDebug() << "Attempting to remove nullptr from square -- aborting.";
//        return;
//    } else {
//        ChessPiece *piece = square->getOccupyingPiece();
//        QGraphicsPixmapItem *sprite = piece->getSprite();
//        if (sprite) {
//            qDebug() << "Removing sprite from square.";
//            chessScene->removeItem(sprite);
//        } else {
//            qDebug() << "Sprite pointer is null.";
//        }
//        square->setOccupyingPiece(nullptr);
//    }
//    return;
//}

///* Alters necessary flag and swaps piece icon */
//void ChessMove::selectPiece(ChessPiece *selectedPiece, ChessSquare *square, QGraphicsScene *chessScene) {
//    selectedPiece->setIsSelected(true);
//    removePieceFromSquare(square, chessScene);
//    addPieceToSquare(selectedPiece, square, 3, chessScene);

//    return;
//}

///* Alters necessary flag and swaps piece icon */
//void ChessMove::deselectPiece(ChessPiece *selectedPiece, ChessSquare *square, QGraphicsScene *chessScene) {
//    selectedPiece->setIsSelected(false);
//    int pieceBaseColor = selectedPiece->getWhite() == true ? 1 : 2;
//    removePieceFromSquare(square, chessScene);
//    addPieceToSquare(selectedPiece, square, pieceBaseColor, chessScene);

//    return;
//}

///* Runs only if square clicked has piece on it! Highlight all potential moves based on the piece selected */
//void ChessMove::highlightPossibleSquares(ChessSquare *square, QGraphicsScene *chessScene, ChessSquare* (&boardSquares)[8][8], bool movePending,
//                                                                        std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares) {
//    ChessPiece *selectedPiece = square->getOccupyingPiece();
//    selectPiece(selectedPiece, square, chessScene);

//    std::vector<int> coords = selectedPiece->getMovesVector();
//    bool lineStopped = false; // Marked true when ecountering a square with friendly piece, since you cannot move through them

//    // Highlight potential moves yellow
//    for (int i = 0; i < (int) coords.size(); i+=2)
//    {
//        int newRank, newFile;
//        int x = coords[i];
//        int y = coords[i+1];

//        // Reset lineStopped when a new direction is started (marked by 0, 0 coords)
//        if (x == 0 && y == 0) {
//            lineStopped = false;
//            continue;
//        }

//        if (lineStopped == false) {
//            if (selectedPiece->getWhite()) {
//                // Light pieces [Currently opponent side]
//                newFile = square->getFile() - x;          // Change in x-axis
//                newRank = square->getRank() + y;     // Change in y-axis
//            } else {
//                // Dark pieces [Currently player side]
//                newFile = square->getFile() + x;          // Change in x-axis
//                newRank = square->getRank() - y;     // Change in y-axis
//            }

//            // Only move forward if coordinate is on the board
//            if (newRank < 8 && newFile < 8 && newRank >= 0 && newFile >= 0) {
//                ChessSquare *possibleMove = boardSquares[newRank][newFile];
//                bool squareOccupied = possibleMove->getOccupyingPiece() == nullptr ? false : true;
//                if (squareOccupied == true) { qDebug()<< "Square occupied by:" << possibleMove->getOccupyingPiece()->getName(); }

//                if (squareOccupied == false) {
//                    // If square is not occupied, it is a potential move
//                    possibleMove->toggleSquareYellow();
//                    highlightedSquares.push_back(possibleMove);
//                    possibleMoveSquares.push_back(possibleMove);
//                    movePending = true;
//                } else if (selectedPiece->getWhite() == true && possibleMove->getOccupyingPiece()->getWhite() == false) {
//                    // If selected piece is white and target piece is black, it is potential move but blocks the line
//                    possibleMove->toggleSquareYellow();
//                    highlightedSquares.push_back(possibleMove);
//                    possibleMoveSquares.push_back(possibleMove);
//                    movePending = true;
//                    lineStopped = true;
//                } else if (selectedPiece->getWhite() == false && possibleMove->getOccupyingPiece()->getWhite() == true) {
//                    // If selected piece is black and target piece is white, it is potential move but blocks the line
//                    possibleMove->toggleSquareYellow();
//                    highlightedSquares.push_back(possibleMove);
//                    possibleMoveSquares.push_back(possibleMove);
//                    movePending = true;
//                    lineStopped = true;
//                } else {
//                    // If square is occupied by friendly, that is blocks the line
//                    lineStopped = true;
//                }
//            }
//        }
//    }
//    return;
//}

///* Deselect piece, empty out highlight and move vectors, reset base square color */
//void ChessMove::movePiece(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare *selectedSquare, bool movePending)
//{
//    ChessPiece *pieceToMove = selectedSquare->getOccupyingPiece(); // Get piece from old square

//    removePieceFromSquare(selectedSquare, chessScene); // Remove sprite from old square
//    deselectPiece(pieceToMove, squareClicked, chessScene); // Add sprite to new square and deselect piece
//    chessScene->update();

//    // Reset all move variables
////    resetPossibleMoveSquares();
////    resetHighlightedSquares();
//    selectedSquare->resetColor();
//    selectedSquare = nullptr;
//    movePending = false;

//    if (pieceToMove->getName() == "Pawn") {
//        Pawn* pawn = dynamic_cast<Pawn*>(pieceToMove);
//        if (pawn->getFirstMove() == true) {
//            pawn->setFirstMove(false);
//        }
//    }

//    return;
//}

//void ChessMove::selectSquare(ChessSquare *squareClicked, QGraphicsScene *chessScene, ChessSquare *selectedSquare, ChessSquare* (&boardSquares)[8][8],
//                                                bool movePending, std::vector<ChessSquare*> &possibleMoveSquares, std::vector<ChessSquare*> &highlightedSquares) {
//    if (selectedSquare != nullptr) {
//        selectedSquare->resetColor();
//    }
//    selectedSquare = squareClicked;
//    squareClicked->toggleSquareYellow();
//    if (squareClicked->getOccupyingPiece() != nullptr) {
//        highlightPossibleSquares(squareClicked, chessScene, boardSquares, movePending, possibleMoveSquares, highlightedSquares);
//    }

//    return;
//}

//void ChessMove::deselectSquare(ChessSquare * squareClicked, ChessSquare *selectedSquare) {
//    if (squareClicked != nullptr) {
//        squareClicked->resetColor();
//    }
//    selectedSquare = nullptr;

//    return;
//}

///* Returns true if the specified chess quare object is a valid place to move */
//bool ChessMove::squareInPossibleMoves(ChessSquare *square, std::vector<ChessSquare*> &possibleMoveSquares)
//{
//    for (ChessSquare *s : possibleMoveSquares) {
//        if (s == square) {
//            return true;
//        }
//    }
//    return false;
//}

///* Reset all highlighted squares to original color */
//void ChessMove::resetHighlightedSquares(std::vector<ChessSquare*> &highlightedSquares) {
//    while (!highlightedSquares.empty()) {
//        ChessSquare *square = highlightedSquares.back();
//        highlightedSquares.pop_back();
//        square->resetColor();
//    }
//    return;
//}

///* Empty the possible moves vector */
//void ChessMove::resetPossibleMoveSquares(std::vector<ChessSquare*> &possibleMoveSquares) {
//    while(!possibleMoveSquares.empty()) {
//        possibleMoveSquares.pop_back();
//    }
//    return;
//}

////void ChessMove::printMoveDebug(QString header) {
////    qDebug() << header;
////    if (selectedSquare != nullptr) { qDebug() << "Selected square:" << selectedSquare->getRank() << selectedSquare->getFile(); } else { qDebug() << "No selected square"; }
////    if (!highlightedSquares.empty()) { qDebug() << "Highlighted squares is NOT empty."; } else { qDebug() << "Highlighted squares is empty."; }
////    if (!possibleMoveSquares.empty()) { qDebug() << "Possible moves is NOT empty"; } else { qDebug() << "Possible moves is empty"; }
////    qDebug() << Qt::endl;

////    return;
////}
