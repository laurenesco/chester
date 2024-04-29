#ifndef PYTHONINTERFACE_H
#define PYTHONINTERFACE_H

#include <QString>
#include <QLabel>

class PythonInterface
{
public:
    PythonInterface();
    QString getEvaluation(QString UCI);
    QString getNextMove(QString UCI);
    void testPython(QLabel *label);
};

#endif // PYTHONINTERFACE_H
