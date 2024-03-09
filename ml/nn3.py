import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nn_parse_pgn import process_pgn_files  # Import the function to process PGN files
import chess

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # Input size is 64 (chess board squares)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output size is 3 (win/loss/draw)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def uci_to_binary(board):
    # Create binary matrix
    binary_matrix = np.zeros((64, 12), dtype=np.uint8)
    # Map pieces to indices
    piece_indices = {
        'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5,
        'r': 6, 'R': 7, 'q': 8, 'Q': 9, 'k': 10, 'K': 11
    }
    # Populate binary matrix
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN:
            binary_matrix[square][piece_indices[piece.symbol().lower()]] = 1
    return binary_matrix.flatten()

# Convert UCI move to one-hot encoding
def uci_to_one_hot(move):
    start_square = ord(move[0]) - ord('a') + (8 - int(move[1])) * 8
    end_square = ord(move[2]) - ord('a') + (8 - int(move[3])) * 8
    return start_square, end_square

# Prepare data for training using nn_parse_pgn.py
X_data, y_data = process_pgn_files("../data")

# Convert data to numpy arrays
X_data = np.array(X_data)
y_data = np.array(y_data)

# Convert UCI board positions to binary matrices
X_data = np.array([uci_to_binary(chess.Board(position)) for position in X_data])

# Convert UCI moves to one-hot encoding
y_data = np.array([uci_to_one_hot(move) for move in y_data])

# Convert data to PyTorch tensors
inputs = torch.tensor(X_data, dtype=torch.float32)
labels = torch.tensor(y_data, dtype=torch.long)

# Print the shape of X_data
print(X_data.shape)

# Create model, loss function, and optimizer
model = ChessNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
batch_size = 32
num_samples = len(inputs)
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        optimizer.zero_grad()
        batch_inputs = inputs[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "chess_model.pth")

