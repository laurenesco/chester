//
// Program Name:              bishop.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the Bishop class. See header file for details.
//

#include "bishop.h"

Bishop::Bishop()
{
    this->icon.load("C://Users//laesc//OneDrive//Desktop//chester//icons//bishop1.png");
}

QPixmap Bishop::getIcon()
{
    return icon;
}

