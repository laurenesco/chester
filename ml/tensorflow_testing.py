import tensorflow as tf
import numpy as np

print('''

*** Begin Program ***

''')

# Generate some sample data
x_train = np.array([1, 2, 3, 4])
y_train = np.array([2, 4, 5, 4])

# Create a linear model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# Compile the model with an optimizer and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=500)

# Make a prediction
print(model.predict(np.array([5])))  # Should output a value close to 5

print('''

*** End Program ***

''')
