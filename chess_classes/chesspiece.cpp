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
    QColor color(22, 38, 48);

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

void ChessPiece::selectPiece()
{

}

int ChessPiece::getPossibleMoves()
{

}

QString ChessPiece::getName()
{
    return m_name;
}

