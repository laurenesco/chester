-- Create table configuration

-- DROP TABLE IF EXISTS public.configuration

CREATE TABLE configuration (
   config_id           	integer     SERIAL PRIMARY KEY,
   config_difficulty 	integer     NOT NULL CHECK (config_difficulty >= 1) CHECK (config_difficulty <= 3),
   config_color 			boolean     NOT NULL,
   config_assist       	boolean     NOT NULL,
);



COMMENT ON TABLE configuration IS 'This table store persistent configuration settings';
COMMENT ON COLUMN configuration.config_id IS 'Unique identifier';
COMMENT ON COLUMN configuration.config_difficulty IS 'Difficulty; 1 - Easy, 2 - Medium, 3 - Hard';
COMMENT ON COLUMN configuration.config_color IS 'Player color; true - white, false - black';
COMMENT ON COLUMN configuration.config_assist IS 'Assist mode set; true - on, false - off';

-- End create table configuration
