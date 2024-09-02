directory organization:


|
|---chess_classes: contains C++ header and source files for the chess related objects
|   |
|   |---piece_classes: contains C++ header and source files for the six chess piece type classes
|
|---data: contains .pgn (portable game notation) files with chess game data for training the machine learning model
|
|---env: contains configuration class, environment variables in a JSON file, and steps to set up the development environment
|
|---icons: contains all of the icons used in the program and applicable attributions
|
|---logos: contains logos (should be combined into icons)
|
|---logs: contains steps in configuring development environment and worklog
|   |
|   |---color palettes: sample color palettes for reference in UX design
|
|---ml: contains all machine learning code
|
|---python: contains the python scripts used in the PythonInterface class
|
|---screens: contains all UI classes
|
|---scripts: contains all SQL scripts needed to create database (postgreSQL)
|
|---stockfish: contains the source code for the stockfish engine, which is currently being used as the engine for the game
|
|---styling: contains style class and Qt qss (css) file
|   |
|   |---color palettes: contains color palettes that may or may not be used for the design of the game
|   |
|   |---fonts: contains fonts used by qss file
