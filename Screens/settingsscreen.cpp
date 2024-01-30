#include "settingsscreen.h"
#include "ui_settingsscreen.h"

// Example query:
/*
QString insertQuery = "INSERT INTO your_table_name (column1, column2) VALUES (?, ?)";
QSqlQuery query;
query.prepare(insertQuery);
query.addBindValue(value1);
query.addBindValue(value2);

if (!query.exec()) {
    qDebug() << "Error executing insert query:" << query.lastError().text();
    return 1;
}
*/

SettingsScreen::SettingsScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SettingsScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Settings");
    setGeometry(200, 85, 1500, 900);

    Config dbConfig;
    dbConfig.configDatabase();
    dbConfig.openDatabase();

    // Execute a query to select all attributes from the table "questions"
    QString queryString = "SELECT game_id, game_winner_color FROM metadata_game";
    QSqlQuery query;

    // Execute a query to select all attributes from the table "questions"
    query.prepare(queryString);
    if (!query.exec()) {
        qWarning() << "Error: Unable to execute query:" << query.lastError().text();
    }

    // Output results to the terminal
    while (query.next()) {
        int questionId = query.value(0).toInt();
        QString question = query.value(1).toString();
        qDebug() << "Game ID:" << questionId << "Winner:" << question;
    }

    dbConfig.closeDatabase();
}

SettingsScreen::~SettingsScreen()
{
    delete ui;
}

void SettingsScreen::on_btn_closeWindow_clicked()
{
    emit settingsScreenClosed();
    this->close();
}
