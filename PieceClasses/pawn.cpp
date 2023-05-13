#include "pawn.h"

Pawn::Pawn()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//pawn1.png");
}

QPixmap Pawn::getIcon()
{
    return icon;
}
