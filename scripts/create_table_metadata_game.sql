-- Create table metadata_game

-- DROP TABLE IF EXISTS game_metadata.metadata_game

CREATE TABLE metadata_game (
	game_id				integer		PRIMARY KEY,
	game_player_color	integer		NOT NULL CHECK (game_player_color >= 0) CHECK (game_difficulty < 2),
	game_winner_color	integer		NOT NULL CHECK (game_winner_color >= 0) CHECK (game_difficulty < 2),,
	game_length			integer		NOT NULL,
	game_moves			varchar(4000),
	game_difficulty		integer		NOT NULL CHECK (game_difficulty >= 0) CHECK (game_difficulty < 4),
	game_date			date		NOT NULL
);



COMMENT ON TABLE metadata_game IS 'This table stoes metadata about completed individual games';
COMMENT ON COLUMN metadata_game.game_id IS 'Unique game identifier';
COMMENT ON COLUMN metadata_game.game_player_color IS 'Players color; 0 - Black, 1 - White';
COMMENT ON COLUMN metadata_game.game_winner_color IS 'Winners color; 0 - Black, 1 - White';
COMMENT ON COLUMN metadata_game.game_length IS 'Length of game in minutes';
COMMENT ON COLUMN metadata_game.game_moves IS 'Algebraic notation of the moves completed';
COMMENT ON COLUMN metadata_game.game_difficulty IS 'Difficulty setting; 0 - Easy, 1 - Standard, 2 - Hard';
COMMENT ON COLUMN metadata_game.game_date IS 'Date played';

-- End create table metadata_game
