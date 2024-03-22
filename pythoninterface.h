#ifndef PYTHONINTERFACE_H
#define PYTHONINTERFACE_H

#include <QString>

class PythonInterface
{
public:
    PythonInterface();
    QString getEvaluation(QString UCI);
};

#endif // PYTHONINTERFACE_H
