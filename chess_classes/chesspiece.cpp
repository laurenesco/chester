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

QPixmap ChessPiece::getSelectedIcon()
{
    QPixmap originalDarkIcon = this->getIcon();
    QPixmap lightIcon = originalDarkIcon.scaled(originalDarkIcon.size());
    QPainter painter(&lightIcon);
    QColor color(255, 182, 193);

    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.fillRect(lightIcon.rect(), color);
    painter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
    painter.drawPixmap(0, 0, originalDarkIcon);
    painter.end();

    return lightIcon;
}

std::vector<int> ChessPiece::getMovesVector()
{
    return movesVector;
}

bool ChessPiece::getIsSelected()
{
    return pieceSelected;
}

QGraphicsPixmapItem *ChessPiece::getSprite()
{
    return sprite;
}

void ChessPiece::setIsSelected(bool value)
{
    pieceSelected = value;
    return;
}

void ChessPiece::setSprite(QGraphicsPixmapItem *sprite)
{
    this->sprite = sprite;
    return;
}

QString ChessPiece::getName()
{
    return name;
}

int ChessPiece::getColor()
{
    return color;
}

void ChessPiece::setColor(int color)
{
    this->color = color;
    return;
}

bool ChessPiece::getWhite() const
{
    return white;
}

void ChessPiece::setWhite(bool newWhite)
{
    white = newWhite;
    if (white) {
        color = 1;
    } else {
        color = 2;
    }
    return;
}

