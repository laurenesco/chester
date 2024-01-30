//
// Program Name:              queen.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the Queen class. Responsible for:
//

#ifndef QUEEN_H
#define QUEEN_H

#include "ChessClasses/chesspiece.h"

class Queen : public ChessPiece {

public:
    Queen();

private:
    QPixmap icon;

    QPixmap getIcon();
};

#endif // QUEEN_H
