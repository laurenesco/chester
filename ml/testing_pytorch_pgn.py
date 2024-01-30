import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import chess  # Library for handling chess data
import chess.pgn  # Module for reading PGN files
import numpy as np
from chess import square

# Load PGN file
pgn_file = open(r"C:\Users\laesc\OneDrive\Desktop\chester\data\bobby1.pgn")

# Create lists to store game data
X_train = []
y_train = []

# Assuming an 8x8x12 board representation
input_features = 8 * 8 * 12

num_epochs = 100

def create_board_representation(game, move):
    """Creates a numerical representation of the board state."""
    board = game.board()
    board_array = np.zeros((8, 8, 12), dtype=np.float32)  # Array for piece positions

    # Map pieces to numerical values
    piece_map = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6}

    for rank in range(8):
        for file in range(8):
            piece = board.piece_at(square(rank, file))
            if piece:
                board_array[rank, file, piece_map[piece.symbol().upper()]] = 1.0

    return board_array.flatten()  # Flatten to a 1D array

def get_outcome(game):
    """Extracts the game outcome from the PGN."""
    root_node = game.root()  # Access the root node
    result = root_node.headers["Result"]
    if result == "1-0":
        return 1  # White win
    elif result == "0-1":
        return 0  # Black win
    else:
        return 0.5  # Draw

for game in chess.pgn.read_game(pgn_file):
    for move in game.mainline_moves():
        board_representation = create_board_representation(game, move)
        outcome = get_outcome(game)  # Extract the outcome

        X_train.append(board_representation)
        y_train.append(outcome)  # Append both board and outcome

# Convert data to PyTorch tensors
X_train = np.array(X_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = np.array(y_train)
y_train = torch.tensor(y_train, dtype=torch.int64)  # Assuming integer labels for outcomes

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer for move probabilities (64 squares)
        self.fc3 = nn.Linear(64, 64)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten input for linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output move probabilities
        x = F.softmax(self.fc3(x), dim=1)
        return x

model = ChessNet()
criterion = nn.CrossEntropyLoss()  # Adjust loss function if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

class ChessDataset(Dataset):
    def __init__(self, data):
        # Load and preprocess your chess data here
        self.data = data

def __len__(self):
    # Returns the total number of samples in the dataset
    return len(self.data)

def __getitem__(self, idx):
    # Retrieves a single sample (board representation, move) by index
    board_rep, move = self.data[idx]
    # Preprocess sample if needed (e.g., convert to tensors)
    return board_rep, move

train_loader = DataLoader(
    dataset=your_chess_dataset,  # Replace with your dataset object
    batch_size=32,  # Adjust batch size as needed
    shuffle=True  # Shuffle data for better training
)

# In the training loop:
for epoch in range(num_epochs):
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
		  # Forward propogation
        model.train()  # Set model to training mode
        output = model(X_train_batch)
        loss = criterion(output, y_train_batch) # Calculate loss based on move predictions and actual moves
		  # Backward propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training progress
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "chess_model.pt")

# Evaluate the model on a test set (implement evaluation code)
