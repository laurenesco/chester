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

            chessSquares[rank][file] = square;
        }
    }
}

void ChessBoard::loadStartingPosition() {
    // Place pawns in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 8; j++) {
            int rank = (i == 0) ? 1 : 6;

            addSpriteToScene("pawn", 5, 5, 10, 10, rank, j);
        }
    }

    // Place rooks in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 0 : 7;

            addSpriteToScene("rook", 5, 5, 10, 10, rank, file);
        }
    }

    // Place knights in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 1 : 6;

            addSpriteToScene("knight", 5, 5, 15, 15, rank, file);
        }
    }

    // Place bishops in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 2 : 5;

            addSpriteToScene("bishop", 5, 5, 10, 10, rank, file);
        }
    }

    // Place kings in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 4 : 4;

            addSpriteToScene("king", 5, 5, 10, 10, rank, file);
          }
    }

    // Place queens in opening position
    for (int i = 0; i < 2; i ++) {
        for (int j = 0; j < 2; j++) {
            int rank = (i == 0) ? 0 : 7;
            int file = (j == 0) ? 3 : 3;

            addSpriteToScene("queen", 5, 5, 10, 10, rank, file);
        }
    }
}

void ChessBoard::addSpriteToScene(QString sprite, int offsetX, int offsetY, int shrinkX, int shrinkY, int rank, int file)
{
    // Create and scale the sprite
    QString spriteName = sprite + "1.png";
    QGraphicsRectItem *squares = chessSquares[rank][file];
    pieceSprite = QPixmap("C://Users//laesc//OneDrive//Desktop//chester//icons//" + spriteName);
    QPixmap scaledPiece = pieceSprite.scaled(tileSize-shrinkX, tileSize-shrinkY, Qt::KeepAspectRatio);
    QGraphicsPixmapItem *finalSprite = new QGraphicsPixmapItem(scaledPiece);

    // Position and add to scene!
    finalSprite->setPos(squares->rect().topLeft() + QPointF(offsetX, offsetY));
    chessScene->addItem(finalSprite);
}

