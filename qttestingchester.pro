# C:\Qt\5.15.0\mingw81_64\bin\qmake.exe

QT += core gui \
                  sql \

# Compiler flags
# QMAKE_CXXFLAGS += -I C:/Users/laesc/anaconda3/include -L C:/Users/laesc/anaconda3/ -lpython311
# QMAKE_CFLAGS += -I C:/Users/laeso/anaconda3/include -L C:/Users/laesc/anaconda3/ -lpython311
QMAKE_CXXFLAGS += -I C:/Users/laesc/AppData/Local/Programs/Python/Python312/include -L C:/Users/laesc/AppData/Local/Programs/Python/Python312 -lpython312
QMAKE_CFLAGS += -I C:/Users/laesc/AppData/Local/Programs/Python/Python312/include -L C:/Users/laesc/AppData/Local/Programs/Python/Python312 -lpython312

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

# LIBS += -LC:/Users/laesc/anaconda3/                  -lpython311
LIBS += -L C:/Users/laesc/AppData/Local/Programs/Python/Python312 -lpython312

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    pythoninterface.cpp \
    screens/boardscreen.cpp \
    screens/helpscreen.cpp \
    screens/settingsscreen.cpp \
    screens/startscreen.cpp \
    styling/colorrepository.cpp \
    chess_classes/chessboard.cpp \
    chess_classes/chessmove.cpp \
    chess_classes/chesspiece.cpp \
    chess_classes/chesssquare.cpp \
    chess_classes/piece_classes/bishop.cpp \
    chess_classes/piece_classes/king.cpp \
    chess_classes/piece_classes/knight.cpp \
    chess_classes/piece_classes/pawn.cpp \
    chess_classes/piece_classes/queen.cpp \
    chess_classes/piece_classes/rook.cpp \
    env/config.cpp \
    main.cpp \

HEADERS += \
    evaluationbar.h \
    pythoninterface.h \
    screens/boardscreen.h \
    screens/helpscreen.h \
    screens/settingsscreen.h \
    screens/startscreen.h \
    styling/colorrepository.h \
    chess_classes/chessboard.h \
    chess_classes/chessmove.h \
    chess_classes/chesspiece.h \
    chess_classes/chesssquare.h \
    chess_classes/piece_classes/bishop.h \
    chess_classes/piece_classes/king.h \
    chess_classes/piece_classes/knight.h \
    chess_classes/piece_classes/pawn.h \
    chess_classes/piece_classes/queen.h \
    chess_classes/piece_classes/rook.h \
    env/config.h \

FORMS += \
    screens/boardscreen.ui \
    screens/helpscreen.ui \
    screens/settingsscreen.ui \
    screens/startscreen.ui \

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    styling/style.qss \
    env/config.json \
    styling/fonts/joystixmonospace.otf
