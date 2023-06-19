#include "chesspiece.h"

// Constructor
ChessPiece::ChessPiece()
{

}

QPixmap ChessPiece::getIcon() {
    return this->getIcon();
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

