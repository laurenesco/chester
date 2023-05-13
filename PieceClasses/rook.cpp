#include "rook.h"

Rook::Rook()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//rook1.png");
}

QPixmap Rook::getIcon()
{
    return icon;
}
