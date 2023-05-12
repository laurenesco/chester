#include "chessboard.h"
#include "qboxlayout.h"

ChessBoard::ChessBoard(QWidget* parent) : QWidget(parent) {
    chessScene = new QGraphicsScene(this);
    chessScene->setBackgroundBrush(Qt::transparent);
    chessView = new QGraphicsView(chessScene, this);
    chessView->setBackgroundBrush(Qt::transparent);
    chessView->setFrameStyle(QFrame::NoFrame);
    chessView->setFixedSize(700, 700);
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(chessView);
    setLayout(layout);

    chessView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    chessView->fitInView(chessScene->sceneRect(), Qt::KeepAspectRatio);

    createChessBoard();
    loadStartingPosition();
}

void ChessBoard::createChessBoard() {
    // Create the visual board
    for (int rank = 0; rank < 8; rank++) {
        for (int file = 0; file < 8; file++) {
            QGraphicsRectItem* square = new QGraphicsRectItem(file * tileSize, rank * tileSize, tileSize, tileSize);
            square->setPen(Qt::NoPen);
            chessScene->addItem(square);

            if ((rank + file) % 2 == 0) {
                square->setBrush(QColor(110, 110, 102));
            } else {
                square->setBrush(QColor(235, 231, 221));
            }

            chessSquares[rank][file] = square; // Store the square items in the array
        }
    }
}

void ChessBoard::loadStartingPosition() {
    // Place pawns in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][j];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//pawn1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-10, tileSize-10, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }

    // Place rooks in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][file];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//rook1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-10, tileSize-10, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }

    // Place knights in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][file];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//knight1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-15, tileSize-15, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }

    // Place bishops in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 2 : 5;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][file];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//bishop1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-10, tileSize-10, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }

    // Place kings in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][file];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//king1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-10, tileSize-10, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }

    // Place queens in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;

            // Create and scale the sprite
            QGraphicsRectItem* squares = chessSquares[rank][file];
            pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//queen1.png");
            QPixmap scaledPiece = pieceSprite.scaled(tileSize-10, tileSize-10, Qt::KeepAspectRatio);
            QGraphicsPixmapItem* sprite = new QGraphicsPixmapItem(scaledPiece);

            // Position and add to scene!
            sprite->setPos(squares->rect().topLeft() + QPointF(5, 5));
            chessScene->addItem(sprite);
        }
    }
}
