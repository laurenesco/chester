import torch
import torch.nn as nn
import torch.nn.functional as F

# Create sample data
X = torch.randn(100, 2)  # 100 samples with 2 features
y = torch.tensor([1 if x1 + x2 > 0 else 0 for x1, x2 in X])  # True if sum of features > 0

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 10)  # Input layer with 2 features, hidden layer with 10 neurons
        self.linear2 = nn.Linear(10, 1)  # Output layer with 1 neuron for binary classification

    def forward(self, x):
        x = F.relu(self.linear1(x))  # Apply ReLU activation
        x = torch.sigmoid(self.linear2(x))  # Apply sigmoid activation for probabilities
        return x

# Create the model instance
model = Net()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

y = y.unsqueeze(1)
y = y.float()

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Make predictions on new data
new_data = torch.tensor([[1.5, -0.5], [-2.0, 1.0]])
predictions = model(new_data)

# Format predictions with labels and percentages
print("Predictions for New Data:")
print("-" * 25)
for i, prediction in enumerate(predictions):
    print(f"Sample {i+1}:")
    print(f"  Probability of Sum of Features > 0: {prediction.item()*100:.2f}%")
