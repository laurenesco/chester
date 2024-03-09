import chess.pgn
import os
import re

# Specify directory containing data
directory = "../data"

# Iterate through each file
for filename in os.listdir(directory):
    if filename.endswith(".pgn"):
        filepath = os.path.join(directory, filename)

        with open(filepath) as pgn_file:

            while True:
                # Read the game from the PGN file
                game = chess.pgn.read_game(pgn_file)

                # Check if there are no more games in the file
                if game is None:
                    break

                # Print game metadata
                print(f"Event: {game.headers.get('Event')}")
                print(f"Site: {game.headers.get('Site')}")
                print(f"Date: {game.headers.get('Date')}")
                print(f"Round: {game.headers.get('Round')}")
                print(f"White: {game.headers.get('White')}")
                print(f"Black: {game.headers.get('Black')}")
                print(f"Result: {game.headers.get('Result')}")
                print(f"ECO: {game.headers.get('ECO')}")
                print(f"White Elo: {game.headers.get('WhiteElo')}")
                print(f"Black Elo: {game.headers.get('BlackElo')}")

                # Print moves
                board = game.board()
                for move in game.mainline_moves():
                    # Skip annotations
                    san_move = re.sub(r"\{.*?\}", "", board.san(move))
                    board.push(move)
                    print(san_move.strip(), end=" ")

                print("\n\n")  # Separate games with empty line

