#include "gamescreen.h"
#include "startscreen.h"

#include <QApplication>
#include <QFile>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Set stylesheet
    QFile file("C:/Users/laesc/OneDrive/Desktop/chester/Styling/style.qss");
    file.open(QFile::ReadOnly);
    QString styleSheet = QLatin1String(file.readAll());
    qApp->setStyleSheet(styleSheet);

    StartScreen s;
    s.show();

    return a.exec();
}
