//
// Program Name:              chesspiece.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Header file for the ChessPiece class. Responsible for:
//

#ifndef CHESSPIECE_H
#define CHESSPIECE_H

#include <QPixmap>
#include <QPainter>
#include <QString>

class ChessPiece
{
public:
    ChessPiece();

    virtual QPixmap getIcon();
    virtual QPixmap getLightIcon();
    virtual QPixmap getDarkIcon();
    virtual void selectPiece();
    virtual std:: vector<int> getMovesVector();
    virtual QString getName();

protected:
    QPixmap m_icon;
    bool m_isSelected;
    QString m_name;
    std:: vector<int> m_movesVector;

private:
};

#endif // CHESSPIECE_H
