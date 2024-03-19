//
// Program Name:              main.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the chesster application.
//

// TODO
//- add pawn capture grid
//- add en passant flag
//- add castling rights
//- add capture logic
//- fine tune ai output
//- implement settings page
//- implement help page
//- add pawn promotion logic
//- move font declaration into styling namespace
//- record game data as intended
//- move tileSize to config file - create configuration class

// - april 5 poster done

#include "screens/startscreen.h"

#include <QApplication>
#include <QFile>

#include <QFontDatabase>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Load custom font from an OTF file
    QFontDatabase::addApplicationFont("C:/Users/laesc/OneDrive/Desktop/chester/styling/fonts/joystixmonospace.otf");

    // Set stylesheet
    QFile file("C:/Users/laesc/OneDrive/Desktop/chester/styling/style.qss");
    file.open(QFile::ReadOnly);
    QString styleSheet = QLatin1String(file.readAll());
    qApp->setStyleSheet(styleSheet);

    StartScreen s;
    s.setWindowState(Qt::WindowMaximized);
    s.show();

    return a.exec();
}
