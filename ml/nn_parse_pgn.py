import os
import chess.pgn
import numpy as np

# Specify the training_directory containing training data
training_directory = "../data"

def encode_board_to_uci(board):
    return board.fen()

def encode_move_to_uci(move):
    return move.uci()

def process_pgn_files(training_directory):
    X_data = []
    y_data = []

    # Iterate over each file in the directory
    for filename in os.listdir(training_directory):
        if filename.endswith(".pgn"):  # Check if the file is a PGN file
            filepath = os.path.join(training_directory, filename)

            # Open the PGN file
            with open(filepath) as pgn_file:
                # Iterate over each game in the PGN file
                while True:
                    # Read the game from the PGN file
                    game = chess.pgn.read_game(pgn_file)

                    # Check if there are no more games in the file
                    if game is None:
                        break

                    # Iterate over each move in the game
                    board = game.board()
                    for move in game.mainline_moves():
                        # Encode the board representation to UCI format
                        encoded_board = encode_board_to_uci(board)
                        # Append the encoded board to the data list
                        X_data.append(encoded_board)
                        # Encode the move to UCI format
                        encoded_move = encode_move_to_uci(move)
                        # Append the encoded move to the labels list
                        y_data.append(encoded_move)
                        # Make the move on the board for the next iteration
                        board.push(move)  
                        
    return X_data, y_data                           

# process_pgn_files(training_directory)

