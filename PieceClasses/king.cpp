//
// Program Name:              king.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the King class. See header file for details.
//

#include "king.h"

King::King()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//king1.png");
}

QPixmap King::getIcon()
{
    return icon;
}
