#include "queen.h"

Queen::Queen()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//queen1.png");
}

QPixmap Queen::getIcon()
{
    return icon;
}
