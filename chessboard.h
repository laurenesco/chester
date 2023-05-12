#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include "sprite.h"
#include <QWidget>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPixmap>

// Functions in the ChessBoard class
//--------------------------------------
// ChessBoard(QWidget *parent)     - Configures the QGraphicsScene and QGraphicsView where the game takes place
// void createChessBoard()               - Generates the gridLayout and Labels which act as the board
// void loadStartingPosition()           - Generates the correct rank and file positions for all the pieces, then calls addSpriteToScene()
// void addSpriteToScene(...)            - Creates, scales, and adds the sprite to the specified location

class ChessBoard : public QWidget {
    Q_OBJECT

public:
    explicit ChessBoard(QWidget* parent = nullptr);

private:
    QGraphicsScene* chessScene;
    QGraphicsView* chessView;
    QGraphicsRectItem* chessSquares[8][8];
    QPixmap pieceSprite;

    int tileSize = 620/8;

    void createChessBoard();
    void loadStartingPosition();
    void addSpriteToScene(QString sprite, int offsetX, int offsetY, int shrinkX, int shrinkY, int rank, int file);
};

#endif // CHESSBOARD_H
