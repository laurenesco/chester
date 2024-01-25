QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    ChessClasses/chessboard.cpp \
    ChessClasses/chesspiece.cpp \
    ChessClasses/chesssquare.cpp \
    PieceClasses/bishop.cpp \
    PieceClasses/king.cpp \
    PieceClasses/knight.cpp \
    PieceClasses/pawn.cpp \
    PieceClasses/queen.cpp \
    PieceClasses/rook.cpp \
    Screens/boardscreen.cpp \
    Screens/helpscreen.cpp \
    Screens/settingsscreen.cpp \
    Screens/startscreen.cpp \
    Styling/colorrepository.cpp \
    logic.cpp \
    main.cpp \

HEADERS += \
    ChessClasses/chessboard.h \
    ChessClasses/chesspiece.h \
    ChessClasses/chesssquare.h \
    PieceClasses/bishop.h \
    PieceClasses/king.h \
    PieceClasses/knight.h \
    PieceClasses/pawn.h \
    PieceClasses/queen.h \
    PieceClasses/rook.h \
    Screens/boardscreen.h \
    Screens/helpscreen.h \
    Screens/settingsscreen.h \
    Screens/startscreen.h \
    Styling/colorrepository.h \
    logic.h \

FORMS += \
    Screens/boardscreen.ui \
    Screens/helpscreen.ui \
    Screens/settingsscreen.ui \
    Screens/startscreen.ui \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    Styling/style.qss
