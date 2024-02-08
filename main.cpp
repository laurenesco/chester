//
// Program Name:              main.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the chesster application.
//

#include "Screens/startscreen.h"

#include <QApplication>
#include <QFile>

#include <QFontDatabase>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Load font family TODO move to styling namespace
    // Load custom font from an OTF file
    int fontId = QFontDatabase::addApplicationFont("C:/Users/laesc/OneDrive/Desktop/chester/Styling/fonts/joystixmonospace.otf");
    if (fontId != -1) {
        QStringList fontFamilies = QFontDatabase::applicationFontFamilies(fontId);
        if (!fontFamilies.isEmpty()) {
            QString fontFamily = fontFamilies.at(0);
            // Use fontFamily in your widget styling
        }
    }

    // Set stylesheet
    QFile file("C:/Users/laesc/OneDrive/Desktop/chester/Styling/style.qss");
    file.open(QFile::ReadOnly);
    QString styleSheet = QLatin1String(file.readAll());
    qApp->setStyleSheet(styleSheet);

    StartScreen s;
    s.setWindowState(Qt::WindowMaximized);
    s.show();

    return a.exec();
}
