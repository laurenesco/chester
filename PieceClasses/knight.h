//
// Program Name:              knight.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Knight class. Responsible for:
//

#ifndef KNIGHT_H
#define KNIGHT_H

#include "ChessClasses/chesspiece.h"

class Knight : public ChessPiece {

public:
    Knight();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // KNIGHT_H
