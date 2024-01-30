//
// Program Name:              config.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     This file handles all global configuration for the application. See header file for more details.
//

#include "config.h"

Config::Config()
{
    m_configPath = "C:/Users/laesc/OneDrive/Desktop/chester/env/config.json";
    db = (QSqlDatabase::addDatabase("QPSQL"));
}

int Config::configDatabase()
{
    qDebug() << Qt::endl << "Entering configDatabase()" << Qt::endl;

    parseJSON();

    // /*
    db.setHostName("localhost");
    db.setPort(5432);
    db.setDatabaseName("chesster");
    db.setUserName("postgres");
    db.setPassword("postgres");
    // */

     /*
    db.setHostName(m_rootObject.value("database").toObject().value("host").toString());
    db.setPort(m_rootObject.value("database").toObject().value("port").toInt());
    db.setDatabaseName(m_rootObject.value("database").toObject().value("name").toString());
    db.setUserName(m_rootObject.value("database").toObject().value("username").toString());
    db.setPassword(m_rootObject.value("database").toObject().value("password").toString());
     */

    qDebug() << Qt::endl << "Leaving configDatabase()" << Qt::endl;

    return 1;
}

int Config::parseJSON() {
    QFile m_configFile(m_configPath);

    // Check for file open error
    if (!m_configFile.open(QIODevice::ReadOnly)) {
        qDebug() << "Failed to open JSON file:" << m_configFile.errorString();
        return 0;
    }

    QByteArray jsonData = m_configFile.readAll();
    m_configFile.close();

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(jsonData, &parseError);
    // Check for json parsing error
    if (doc.isNull()) {
        qDebug() << "Failed to parse JSON:" << parseError.errorString();
        return 0;
    }

    // Check for object error
    if (!doc.isObject()) {
        qDebug() << "JSON document is not an object";
        return 0;
    }

    m_rootObject = doc.object();

    return 1;
}

int Config::openDatabase()
{
    bool connectioncheck = db.open();

    if (connectioncheck == true){
        qDebug() << "Connection to database established." << Qt::endl;
        return 1;
    } else {
        qDebug() << QSqlDatabase::drivers() << Qt::endl;
        qDebug() << "Error opening database " << db.databaseName() << " :" << db.lastError().text() << Qt::endl;
        return 0;
    }
}

int Config::closeDatabase()
{
    if (db.isOpen()) {
        db.close();
        qDebug() << "Database closed.";
        return 1;
    } else {
        qDebug() << "No database open.";
        return 0;
    }
}
