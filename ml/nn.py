import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nn_parse_pgn import process_pgn_files  # Import the function to process PGN files

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

# Prepare data for training using nn_parse_pgn.py
X_data, y_data = process_pgn_files("../data")
X_data = np.array(X_data)
y_data = np.array(y_data)

# Convert data to PyTorch tensors
print(X_data.shape)
inputs = torch.tensor(X_data, dtype=torch.float32)
labels = torch.tensor(y_data, dtype=torch.long)

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

