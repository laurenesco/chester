#include "bishop.h"

Bishop::Bishop()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//bishop1.png");
}

QPixmap Bishop::getIcon()
{
    return icon;
}

