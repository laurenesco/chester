/********************************************************************************
** Form generated from reading UI file 'gamescreen.ui'
**
** Created by: Qt User Interface Compiler version 5.15.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GAMESCREEN_H
#define UI_GAMESCREEN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QVBoxLayout *verticalLayout;
    QFrame *frmContainer;
    QVBoxLayout *verticalLayout_2;
    QFrame *frmEngine;
    QFrame *frmMain;
    QHBoxLayout *horizontalLayout;
    QFrame *frmGame;
    QVBoxLayout *verticalLayout_3;
    QFrame *frmBoard;
    QFrame *frmMoves;
    QFrame *frmStats;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1202, 1001);
        MainWindow->setMinimumSize(QSize(0, 0));
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        verticalLayout = new QVBoxLayout(centralwidget);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frmContainer = new QFrame(centralwidget);
        frmContainer->setObjectName(QString::fromUtf8("frmContainer"));
        frmContainer->setFrameShape(QFrame::StyledPanel);
        frmContainer->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frmContainer);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        verticalLayout_2->setContentsMargins(0, 0, 0, 0);
        frmEngine = new QFrame(frmContainer);
        frmEngine->setObjectName(QString::fromUtf8("frmEngine"));
        QSizePolicy sizePolicy(QSizePolicy::Maximum, QSizePolicy::Maximum);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frmEngine->sizePolicy().hasHeightForWidth());
        frmEngine->setSizePolicy(sizePolicy);
        frmEngine->setMinimumSize(QSize(1178, 57));
        frmEngine->setMaximumSize(QSize(1178, 57));
        frmEngine->setFrameShape(QFrame::StyledPanel);
        frmEngine->setFrameShadow(QFrame::Raised);

        verticalLayout_2->addWidget(frmEngine);

        frmMain = new QFrame(frmContainer);
        frmMain->setObjectName(QString::fromUtf8("frmMain"));
        frmMain->setFrameShape(QFrame::StyledPanel);
        frmMain->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frmMain);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        frmGame = new QFrame(frmMain);
        frmGame->setObjectName(QString::fromUtf8("frmGame"));
        frmGame->setMinimumSize(QSize(868, 861));
        frmGame->setMaximumSize(QSize(868, 861));
        frmGame->setFrameShape(QFrame::StyledPanel);
        frmGame->setFrameShadow(QFrame::Raised);
        verticalLayout_3 = new QVBoxLayout(frmGame);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        verticalLayout_3->setContentsMargins(0, 0, 0, 0);
        frmBoard = new QFrame(frmGame);
        frmBoard->setObjectName(QString::fromUtf8("frmBoard"));
        sizePolicy.setHeightForWidth(frmBoard->sizePolicy().hasHeightForWidth());
        frmBoard->setSizePolicy(sizePolicy);
        frmBoard->setMinimumSize(QSize(700, 700));
        frmBoard->setMaximumSize(QSize(700, 700));
        frmBoard->setFrameShape(QFrame::StyledPanel);
        frmBoard->setFrameShadow(QFrame::Raised);

        verticalLayout_3->addWidget(frmBoard, 0, Qt::AlignHCenter|Qt::AlignVCenter);

        frmMoves = new QFrame(frmGame);
        frmMoves->setObjectName(QString::fromUtf8("frmMoves"));
        sizePolicy.setHeightForWidth(frmMoves->sizePolicy().hasHeightForWidth());
        frmMoves->setSizePolicy(sizePolicy);
        frmMoves->setMinimumSize(QSize(868, 150));
        frmMoves->setMaximumSize(QSize(868, 150));
        frmMoves->setFrameShape(QFrame::StyledPanel);
        frmMoves->setFrameShadow(QFrame::Raised);

        verticalLayout_3->addWidget(frmMoves, 0, Qt::AlignHCenter|Qt::AlignBottom);

        verticalLayout_3->setStretch(1, 1);

        horizontalLayout->addWidget(frmGame, 0, Qt::AlignHCenter);

        frmStats = new QFrame(frmMain);
        frmStats->setObjectName(QString::fromUtf8("frmStats"));
        sizePolicy.setHeightForWidth(frmStats->sizePolicy().hasHeightForWidth());
        frmStats->setSizePolicy(sizePolicy);
        frmStats->setMinimumSize(QSize(300, 861));
        frmStats->setMaximumSize(QSize(300, 861));
        frmStats->setFrameShape(QFrame::StyledPanel);
        frmStats->setFrameShadow(QFrame::Raised);

        horizontalLayout->addWidget(frmStats);

        horizontalLayout->setStretch(0, 6);
        horizontalLayout->setStretch(1, 3);

        verticalLayout_2->addWidget(frmMain);

        verticalLayout_2->setStretch(0, 1);
        verticalLayout_2->setStretch(1, 15);

        verticalLayout->addWidget(frmContainer);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        menubar->setGeometry(QRect(0, 0, 1202, 25));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GAMESCREEN_H
