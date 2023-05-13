#include "sprite.h"

Sprite::Sprite()
{
    // Load all sprites
    pawn.load("C://Users//laesc//OneDrive//Desktop//chester//icons//pawn1.png");
    rook.load("C://Users//laesc//OneDrive//Desktop//chester//icons//rook1.png");
    knight.load("C://Users//laesc//OneDrive//Desktop//chester//icons//knight1.png");
    bishop.load("C://Users//laesc//OneDrive//Desktop//chester//icons//bishop1.png");
    queen.load("C://Users//laesc//OneDrive//Desktop//chester//icons//queen1.png");
    king.load("C://Users//laesc//OneDrive//Desktop//chester//icons//king1.png");
}

// Get method for all sprites
QPixmap Sprite::getPawn() { return pawn; }
QPixmap Sprite::getBishop() { return bishop; }
QPixmap Sprite::getRook() { return rook; }
QPixmap Sprite::getKnight() { return knight; }
QPixmap Sprite::getKing() { return king; }
QPixmap Sprite::getQueen() { return queen; }

