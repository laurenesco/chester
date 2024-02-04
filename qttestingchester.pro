# C:\Qt\5.15.0\mingw81_64\bin\qmake.exe

QT += core gui \
                  sql \

# Compiler flags
QMAKE_CXXFLAGS += -I C:/Users/laesc/AppData/Local/Programs/Python/Python310/include -L C:/Users/laesc/AppData/Local/Programs/Python/Python310/libs -lpython310
QMAKE_CFLAGS += -I C:/Users/laesc/AppData/Local/Programs/Python/Python310/include -L C:/Users/laesc/AppData/Local/Programs/Python/Python310/libs -lpython310

# Python.h config, Python.h uses the variable slots, so that cannot be reserve word for Qt anymore. Consequently,
# these changes must be made in all code written: signals -> Q_SIGNALS, slots   -> Q_SLOTS, emit    -> Q_EMIT
# See https://stackoverflow.com/questions/15078060/embedding-python-in-qt-5
CONFIG += no_keywords

# Include Path
INCLUDEPATH += "C:/pgsql/include"

# Library Paths
LIBS += -L"C:/pgsql/lib" \
             -L"C:/Qt/5.15.0/mingw81_64/plugins/sqldrivers" \
             -L"C:/Qt/5.15.0/Src/qtbase/src/plugins/sqldrivers/psql" \
             -lqsqlpsql

LIBS += -LC:/Users/laesc/AppData/Local/Programs/Python/Python310/libs -lpython310

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
    env/config.cpp \
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
    env/config.h \
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
    Styling/style.qss \
    env/config.json
