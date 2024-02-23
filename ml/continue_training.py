import torch
import torch.nn as nn
import torch.optim as optim
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

print("Pre-trained model loaded...")

# Convert UCI move to one-hot encoding
def uci_to_one_hot(move):
    start_square = ord(move[0]) - ord('a') + (8 - int(move[1])) * 8
    end_square = ord(move[2]) - ord('a') + (8 - int(move[3])) * 8
    return start_square, end_square

print("UCI -> One hot encoding conversion defined...")

# Load data generator
data_generator = process_pgn_files("../data")

print("Data generator created...")

# Create loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Beginning training...")

# Train the model
num_epochs = 1  # Changed to 1 for testing, was 10
for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    num_batches = 0
    total_loss = 0
    move_counter = 0
    for board, move in data_generator:
        print(f"{epoch} - {move_counter} - Starting forward pass...")
        # Convert UCI board position to binary matrix
        def uci_to_binary(board):
            binary_matrix = np.zeros((64, 12), dtype=np.uint8)
            piece_indices = {'p': 0, 'P': 1, 'n': 2, 'N': 3, 'b': 4, 'B': 5, 'r': 6, 'R': 7, 'q': 8, 'Q': 9, 'k': 10, 'K': 11}
            for square, piece in board.piece_map().items():
                if piece.piece_type != chess.PAWN:
                    binary_matrix[square][piece_indices[piece.symbol().lower()]] = 1
            return binary_matrix.flatten()
        X_data = uci_to_binary(chess.Board(board))
        X_data = torch.tensor(X_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        # Convert UCI move to one-hot encoding
        y_data = uci_to_one_hot(move)
        y_data = torch.tensor(y_data[1], dtype=torch.long).unsqueeze(0)  # Add batch dimension
        print(f"{epoch} - {move_counter} - Starting backpropagation...")
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        num_batches += 1
        total_loss += loss.item()
        move_counter += 1
    average_loss = total_loss / num_batches
    print(f"  Average Loss: {average_loss:.4f}")

# Save the updated model
torch.save(model.state_dict(), "chess_model.pth")

print("Training completed. Updated model saved.")

