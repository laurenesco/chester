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

private:
    QJsonObject m_rootObject;
    QString m_configPath;
    QFile m_configFile;
    QString m_dbhost;
    QString m_dbName;
    QString m_dbusername;
    QString m_dbPassword;
    int m_dbPort;
};

#endif // CONFIG_H
