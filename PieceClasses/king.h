//
// Program Name:              king.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the King class. Responsible for:
//

#ifndef KING_H
#define KING_H

#include "ChessClasses/chesspiece.h"

class King : public ChessPiece {

public:
    King();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // KING_H
