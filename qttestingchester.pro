QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    PieceClasses/bishop.cpp \
    PieceClasses/king.cpp \
    PieceClasses/knight.cpp \
    PieceClasses/pawn.cpp \
    PieceClasses/queen.cpp \
    PieceClasses/rook.cpp \
    chessboard.cpp \
    chesspiece.cpp \
    chesssquare.cpp \
    logic.cpp \
    main.cpp \
    mainwindow.cpp \
    startscreen.cpp

HEADERS += \
    PieceClasses/bishop.h \
    PieceClasses/king.h \
    PieceClasses/knight.h \
    PieceClasses/pawn.h \
    PieceClasses/queen.h \
    PieceClasses/rook.h \
    chessboard.h \
    chesspiece.h \
    chesssquare.h \
    logic.h \
    mainwindow.h \
    startscreen.h

FORMS += \
    mainwindow.ui \
    startscreen.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
