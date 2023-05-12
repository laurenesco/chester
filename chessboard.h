#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include "sprites.h"
#include <QWidget>
#include <QGraphicsRectItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPixmap>

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
};

#endif // CHESSBOARD_H
