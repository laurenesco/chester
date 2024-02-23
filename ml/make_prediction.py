import torch
import torch.nn as nn
import numpy as np
import chess
from nn_parse_pgn import process_pgn_files  # Import the function to process PGN files

print("Starting program...")

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(768, 128)  # Input size is 768 (binary matrix size)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)  # Output size is 64 (one-hot encoding size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the pre-trained model
model = ChessNet()
model.load_state_dict(torch.load("chess_model.pth"))

print("ChessNet defined and pre-trained model loaded...")

def index_to_algebraic(index):
    start_square = index // 64
    end_square = index % 64
    start_square_coords = chess.square_name(start_square)
    end_square_coords = chess.square_name(end_square)
    return start_square_coords + end_square_coords

# Convert UCI move to one-hot encoding
def uci_to_one_hot(move):
    start_square = ord(move[0]) - ord('a') + (8 - int(move[1])) * 8
    end_square = ord(move[2]) - ord('a') + (8 - int(move[3])) * 8
    return start_square, end_square

print("UCI -> One hot encoding conversion defined...")

# Get user input for board representation in UCI format
board_position = input("Enter the board position in UCI format: ")

# Convert UCI board position to binary matrix
def uci_to_binary(board):
    binary_matrix = np.zeros((64, 12), dtype=np.uint8)
    piece_indices = {'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5, 'r': 6, 'R': 7, 'q': 8, 'Q': 9, 'k': 10, 'K': 11}
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            binary_matrix[square][piece_indices[piece.symbol().lower()]] = 1
    return binary_matrix.flatten()

X_data = uci_to_binary(chess.Board(board_position))
X_data = torch.tensor(X_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Make predictions
outputs = model(X_data)
predicted_move_index = torch.argmax(outputs)
print(f"Predicted Move Index: {predicted_move_index}")

predicted_move = index_to_algebraic(predicted_move_index.item())
print(f"Predicted Move: {predicted_move}")

print("Prediction process completed.")

