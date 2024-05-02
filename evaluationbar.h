#ifndef EVALUATIONBAR_H
#define EVALUATIONBAR_H

#include <QPainter>
#include <QWidget>

// Update using syntax: bar->setEvaluation(-75.0);

class EvaluationBar : public QWidget
{
    Q_OBJECT
public:
    explicit EvaluationBar(QWidget *parent = nullptr) : QWidget(parent), evaluation(0.0) { }

    void setEvaluation(double eval) {
        evaluation = eval;
        update();  // Triggers a repaint
    }

protected:
    double scaleEvaluation(double eval) {
        const double scale = 100.0;  // Determines the scaling factor
        return scale * (std::log(std::abs(eval) + 1) / std::log(301));  // Log base change to scale from -300 to +300
    }

    void paintEvent(QPaintEvent *event) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.fillRect(rect(), Qt::white);

        // Adjust the evaluation using logarithmic scaling
        double adjustedEval = scaleEvaluation(evaluation);

        // Now map the adjusted evaluation to the bar width
        double blackPortion = (adjustedEval + 100.0) / 200.0;
        int blackWidth = static_cast<int>(width() * blackPortion);

        painter.fillRect(0, 0, blackWidth, height(), Qt::black);
    }

private:
    double evaluation;  // Store the evaluation from -100 to +100
};

#endif // EVALUATIONBAR_H
