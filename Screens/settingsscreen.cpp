//
// Program Name:              settingsscreen.cpp
// Date Last Modified:        01/30/2024
// Last Modified By:            Lauren Escobedo
//
// Program Description:     Driver file for the SettingsScreen class. See header file for details.
//

#include "settingsscreen.h"
#include "ui_settingsscreen.h"

// Constructor
SettingsScreen::SettingsScreen(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SettingsScreen)
{
    // Setup the mainwindow
    ui->setupUi(this);
    setWindowTitle("chesster - Settings");

    fillComboBoxes();
}

// Deconstructor
SettingsScreen::~SettingsScreen()
{
    delete ui;
}

void SettingsScreen::fillComboBoxes()
{
    // Difficulty combobox
    QStringList list = {"Easy", "Medium", "Hard"};
    ui->cmb_difficulty->addItems(list);

    // Player color
    list = {"White", "Black"};
    ui->cmb_color->addItems(list);

    // Assisted mode
    list = {"On", "Off"};
    ui->cmb_assist->addItems(list);
}

// On pressing Close button, emit closing signal and close this form
void SettingsScreen::on_btn_closeWindow_clicked()
{
    Q_EMIT settingsScreenClosed();
    this->close();
}

// sql code:
// Set up the database connection
//Config dbConfig;
//dbConfig.configDatabase();
//dbConfig.openDatabase();

//// Execute a query to select all attributes from the table "questions"
//QString queryString = "SELECT game_id, game_winner_color FROM metadata_game";
//QSqlQuery query;
//query.prepare(queryString);
//if (!query.exec()) {
//    qWarning() << "Error: Unable to execute query:" << query.lastError().text();
//}

//// Output results to the terminal
//while (query.next()) {
//    int questionId = query.value(0).toInt();
//    QString question = query.value(1).toString();
//    qDebug() << "Game ID:" << questionId << "Winner:" << question;
//}

//query.clear();
//dbConfig.closeDatabase();

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
