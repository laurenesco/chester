/********************************************************************************
** Form generated from reading UI file 'startscreen.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_STARTSCREEN_H
#define UI_STARTSCREEN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_StartScreen
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout_2;
    QFrame *frm_main_upper;
    QFrame *frm_main_logo;
    QVBoxLayout *verticalLayout;
    QLabel *lbl_logo;
    QFrame *frm_main_lower;
    QVBoxLayout *verticalLayout_6;
    QFrame *frm_start;
    QHBoxLayout *horizontalLayout;
    QFrame *frame;
    QFrame *frame_2;
    QVBoxLayout *verticalLayout_5;
    QPushButton *btn_play;
    QFrame *frame_3;
    QFrame *frm_settings;
    QHBoxLayout *horizontalLayout_2;
    QFrame *frame_6;
    QFrame *frame_5;
    QVBoxLayout *verticalLayout_4;
    QPushButton *btn_settings;
    QFrame *frame_4;
    QFrame *frm_help;
    QHBoxLayout *horizontalLayout_3;
    QFrame *frame_9;
    QFrame *frame_8;
    QVBoxLayout *verticalLayout_3;
    QPushButton *btn_help;
    QFrame *frame_7;
    QFrame *frame_11;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *StartScreen)
    {
        if (StartScreen->objectName().isEmpty())
            StartScreen->setObjectName(QString::fromUtf8("StartScreen"));
        StartScreen->resize(1500, 900);
        centralwidget = new QWidget(StartScreen);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout_2 = new QVBoxLayout(centralwidget);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        frm_main_upper = new QFrame(centralwidget);
        frm_main_upper->setObjectName(QString::fromUtf8("frm_main_upper"));
        frm_main_upper->setFrameShape(QFrame::StyledPanel);
        frm_main_upper->setFrameShadow(QFrame::Raised);

        verticalLayout_2->addWidget(frm_main_upper);

        frm_main_logo = new QFrame(centralwidget);
        frm_main_logo->setObjectName(QString::fromUtf8("frm_main_logo"));
        frm_main_logo->setFrameShape(QFrame::StyledPanel);
        frm_main_logo->setFrameShadow(QFrame::Raised);
        verticalLayout = new QVBoxLayout(frm_main_logo);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        lbl_logo = new QLabel(frm_main_logo);
        lbl_logo->setObjectName(QString::fromUtf8("lbl_logo"));
        lbl_logo->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(lbl_logo);


        verticalLayout_2->addWidget(frm_main_logo);

        frm_main_lower = new QFrame(centralwidget);
        frm_main_lower->setObjectName(QString::fromUtf8("frm_main_lower"));
        frm_main_lower->setFrameShape(QFrame::StyledPanel);
        frm_main_lower->setFrameShadow(QFrame::Raised);
        verticalLayout_6 = new QVBoxLayout(frm_main_lower);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        frm_start = new QFrame(frm_main_lower);
        frm_start->setObjectName(QString::fromUtf8("frm_start"));
        frm_start->setFrameShape(QFrame::StyledPanel);
        frm_start->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frm_start);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(-1, 0, -1, 0);
        frame = new QFrame(frm_start);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);

        horizontalLayout->addWidget(frame);

        frame_2 = new QFrame(frm_start);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        verticalLayout_5 = new QVBoxLayout(frame_2);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        verticalLayout_5->setContentsMargins(-1, 5, -1, 5);
        btn_play = new QPushButton(frame_2);
        btn_play->setObjectName(QString::fromUtf8("btn_play"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(btn_play->sizePolicy().hasHeightForWidth());
        btn_play->setSizePolicy(sizePolicy);
        btn_play->setBaseSize(QSize(0, 0));
        QFont font;
        font.setPointSize(12);
        btn_play->setFont(font);
        btn_play->setStyleSheet(QString::fromUtf8(""));

        verticalLayout_5->addWidget(btn_play);


        horizontalLayout->addWidget(frame_2);

        frame_3 = new QFrame(frm_start);
        frame_3->setObjectName(QString::fromUtf8("frame_3"));
        frame_3->setFrameShape(QFrame::StyledPanel);
        frame_3->setFrameShadow(QFrame::Raised);

        horizontalLayout->addWidget(frame_3);

        horizontalLayout->setStretch(0, 2);
        horizontalLayout->setStretch(1, 3);
        horizontalLayout->setStretch(2, 2);

        verticalLayout_6->addWidget(frm_start);

        frm_settings = new QFrame(frm_main_lower);
        frm_settings->setObjectName(QString::fromUtf8("frm_settings"));
        frm_settings->setFrameShape(QFrame::StyledPanel);
        frm_settings->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frm_settings);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(-1, 0, -1, 0);
        frame_6 = new QFrame(frm_settings);
        frame_6->setObjectName(QString::fromUtf8("frame_6"));
        frame_6->setFrameShape(QFrame::StyledPanel);
        frame_6->setFrameShadow(QFrame::Raised);

        horizontalLayout_2->addWidget(frame_6);

        frame_5 = new QFrame(frm_settings);
        frame_5->setObjectName(QString::fromUtf8("frame_5"));
        frame_5->setFrameShape(QFrame::StyledPanel);
        frame_5->setFrameShadow(QFrame::Raised);
        verticalLayout_4 = new QVBoxLayout(frame_5);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        verticalLayout_4->setContentsMargins(-1, 5, -1, 5);
        btn_settings = new QPushButton(frame_5);
        btn_settings->setObjectName(QString::fromUtf8("btn_settings"));
        sizePolicy.setHeightForWidth(btn_settings->sizePolicy().hasHeightForWidth());
        btn_settings->setSizePolicy(sizePolicy);
        btn_settings->setFont(font);

        verticalLayout_4->addWidget(btn_settings);


        horizontalLayout_2->addWidget(frame_5);

        frame_4 = new QFrame(frm_settings);
        frame_4->setObjectName(QString::fromUtf8("frame_4"));
        frame_4->setFrameShape(QFrame::StyledPanel);
        frame_4->setFrameShadow(QFrame::Raised);

        horizontalLayout_2->addWidget(frame_4);

        horizontalLayout_2->setStretch(0, 2);
        horizontalLayout_2->setStretch(1, 3);
        horizontalLayout_2->setStretch(2, 2);

        verticalLayout_6->addWidget(frm_settings);

        frm_help = new QFrame(frm_main_lower);
        frm_help->setObjectName(QString::fromUtf8("frm_help"));
        frm_help->setFrameShape(QFrame::StyledPanel);
        frm_help->setFrameShadow(QFrame::Raised);
        horizontalLayout_3 = new QHBoxLayout(frm_help);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(-1, 0, -1, 0);
        frame_9 = new QFrame(frm_help);
        frame_9->setObjectName(QString::fromUtf8("frame_9"));
        frame_9->setFrameShape(QFrame::StyledPanel);
        frame_9->setFrameShadow(QFrame::Raised);

        horizontalLayout_3->addWidget(frame_9);

        frame_8 = new QFrame(frm_help);
        frame_8->setObjectName(QString::fromUtf8("frame_8"));
        frame_8->setFrameShape(QFrame::StyledPanel);
        frame_8->setFrameShadow(QFrame::Raised);
        verticalLayout_3 = new QVBoxLayout(frame_8);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(11, 5, -1, 5);
        btn_help = new QPushButton(frame_8);
        btn_help->setObjectName(QString::fromUtf8("btn_help"));
        sizePolicy.setHeightForWidth(btn_help->sizePolicy().hasHeightForWidth());
        btn_help->setSizePolicy(sizePolicy);
        btn_help->setFont(font);

        verticalLayout_3->addWidget(btn_help);


        horizontalLayout_3->addWidget(frame_8);

        frame_7 = new QFrame(frm_help);
        frame_7->setObjectName(QString::fromUtf8("frame_7"));
        frame_7->setFrameShape(QFrame::StyledPanel);
        frame_7->setFrameShadow(QFrame::Raised);

        horizontalLayout_3->addWidget(frame_7);

        horizontalLayout_3->setStretch(0, 2);
        horizontalLayout_3->setStretch(1, 3);
        horizontalLayout_3->setStretch(2, 2);

        verticalLayout_6->addWidget(frm_help);

        frame_11 = new QFrame(frm_main_lower);
        frame_11->setObjectName(QString::fromUtf8("frame_11"));
        frame_11->setFrameShape(QFrame::StyledPanel);
        frame_11->setFrameShadow(QFrame::Raised);

        verticalLayout_6->addWidget(frame_11);

        verticalLayout_6->setStretch(0, 2);
        verticalLayout_6->setStretch(1, 2);
        verticalLayout_6->setStretch(2, 2);
        verticalLayout_6->setStretch(3, 5);

        verticalLayout_2->addWidget(frm_main_lower);

        verticalLayout_2->setStretch(0, 2);
        verticalLayout_2->setStretch(1, 4);
        verticalLayout_2->setStretch(2, 6);
        StartScreen->setCentralWidget(centralwidget);
        menubar = new QMenuBar(StartScreen);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1500, 25));
        StartScreen->setMenuBar(menubar);
        statusbar = new QStatusBar(StartScreen);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        StartScreen->setStatusBar(statusbar);

        retranslateUi(StartScreen);

        QMetaObject::connectSlotsByName(StartScreen);
    } // setupUi

    void retranslateUi(QMainWindow *StartScreen)
    {
        StartScreen->setWindowTitle(QCoreApplication::translate("StartScreen", "MainWindow", nullptr));
        lbl_logo->setText(QString());
        btn_play->setText(QCoreApplication::translate("StartScreen", "Play", nullptr));
        btn_settings->setText(QCoreApplication::translate("StartScreen", "Settings", nullptr));
        btn_help->setText(QCoreApplication::translate("StartScreen", "Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class StartScreen: public Ui_StartScreen {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_STARTSCREEN_H
