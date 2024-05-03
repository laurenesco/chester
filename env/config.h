//
// Program Name:              config.h
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     This file handles all global configuration for the application. This includes:
//                                              - Database configuration
//

#ifndef CONFIG_H
#define CONFIG_H

#include <QSqlDatabase>
#include <QSqlDriver>
#include <QSqlDriverPlugin>
#include <QSqlQuery>
#include <QSqlError>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

#include <QDebug>

class Config
{
public:
    Config();

    QSqlDatabase db;

    int configDatabase();
    int openDatabase();
    int closeDatabase();
    int parseJSON();
    void refreshConfig();
    void saveConfig();

    int getDifficulty() const;
    void setDifficulty(int newDifficulty);

    bool getColor() const;
    void setColor(bool newColor);

    bool getAssistModeOn() const;
    void setAssistModeOn(bool newAssistModeOn);

private:
    // Database members
    QJsonObject m_rootObject;
    QString m_configPath;
    QFile m_configFile;
    QString m_dbhost;
    QString m_dbName;
    QString m_dbusername;
    QString m_dbPassword;
    int m_dbPort;

    // Settings members
    int difficulty = 1;
    bool color = true;
    bool assistModeOn = true;
};

#endif // CONFIG_H
