#include "sprites.h"

sprites::sprites()
{
    // Load all sprites
    pawn.load("C://Users//laesc//OneDrive//Desktop//chester//icons//pawn1.png");
    rook.load("..//icons//rook.png");
    knight.load("..//icons//knight1.png");
    bishop.load("..//icons//bishop1.png");
    queen.load("..//icons//queen1.png");
    king.load("..//icons//king1.png");
}

// Get method for all sprites
QPixmap sprites::getPawn() { return pawn; }
QPixmap sprites::getBishop() { return bishop; }
QPixmap sprites::getRook() { return rook; }
QPixmap sprites::getKnight() { return knight; }
QPixmap sprites::getKing() { return king; }
QPixmap sprites::getQueen() { return queen; }

