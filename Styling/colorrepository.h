//
// Program Name:              logic.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the logic class. Responsible for:
//

#ifndef COLORREPOSITORY_H
#define COLORREPOSITORY_H

#include <QColor>
#include <QPalette>

namespace colorrepository {
    QPalette standardPalette();
    void setDarkMode(bool dark);

    QColor windowBackground();
    QColor baseBackground();
    QColor text();

    QColor pressedTextColor();
    QColor hoverTextColor();

    QColor pressedOutlineColor();
    QBrush hoverOutlineBrush(const QRect &rect);
    QColor buttonOutlineColor();

    QColor buttonPressedBackground();
    QColor buttonHoveredBackground();
    QColor buttonBackground();

    QBrush progressBarOutlineBrush(const QRect &rect);
    QBrush progressBarOutlineFadingBrush(const QRect &rect);
    QBrush progressBarContentsBrush(const QRect &rect);
    QColor progressBarTextColor(bool enabled);
}

#endif // COLORREPOSITORY_H
