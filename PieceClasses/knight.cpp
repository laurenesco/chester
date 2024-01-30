//
// Program Name:              knight.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Knight class. See header file for details.
//

#include "knight.h"

Knight::Knight()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//knight1.png");
}

QPixmap Knight::getIcon()
{
    return icon;
}
