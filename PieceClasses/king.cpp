#include "king.h"

King::King()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//king1.png");
}

QPixmap King::getIcon()
{
    return icon;
}
