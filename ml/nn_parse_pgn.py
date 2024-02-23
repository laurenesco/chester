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
                        # Encode the move to UCI format
                        encoded_move = encode_move_to_uci(move)
                        # Make the move on the board for the next iteration
                        board.push(move)
                        # Yield the encoded board and move
                        print (encoded_board)
                        yield encoded_board, encoded_move


# for encoded_board, encoded_move in process_pgn_files(training_directory):
#     print(encoded_board)
