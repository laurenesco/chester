#include "knight.h"

Knight::Knight()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//knight1.png");
}

QPixmap Knight::getIcon()
{
    return icon;
}
