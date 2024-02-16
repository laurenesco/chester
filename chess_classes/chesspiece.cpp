//
// Program Name:              chesspiece.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the ChessPiece class. See header file for details.
//

#include "chesspiece.h"

// Constructor
ChessPiece::ChessPiece()
{

}

QPixmap ChessPiece::getIcon() {
    return icon;
}

QPixmap ChessPiece::getLightIcon()
{
    QPixmap originalDarkIcon = this->getIcon();
    QPixmap lightIcon = originalDarkIcon.scaled(originalDarkIcon.size());
    QPainter painter(&lightIcon);
    QColor color(255, 255, 255);

    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(lightIcon.rect(), color);
    painter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
    painter.drawPixmap(0, 0, originalDarkIcon);
    painter.end();

    return lightIcon;
}

QPixmap ChessPiece::getDarkIcon()
{
    return this->getIcon();
}

std::vector<int> ChessPiece::getMovesVector()
{
    return movesVector;
}

bool ChessPiece::getIsSelected()
{
    return isSelected;
}

void ChessPiece::setIsSelected(bool value)
{
    isSelected = value;
    return;
}

QString ChessPiece::getName()
{
    return name;
}

bool ChessPiece::getColor()
{
    return isWhite;
}

void ChessPiece::setColor(bool value)
{
    isWhite = value;
    return;
}

