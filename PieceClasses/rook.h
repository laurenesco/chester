//
// Program Name:              rook.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Rook class. Responsible for:
//

#ifndef ROOK_H
#define ROOK_H

#include "ChessClasses/chesspiece.h"

class Rook : public ChessPiece {

public:
    Rook();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // ROOK_H
