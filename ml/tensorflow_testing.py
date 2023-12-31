import tensorflow as tf

# Generate some sample data
x_train = [1, 2, 3, 4]
y_train = [2, 4, 5, 4]

# Create a linear model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model with an optimizer and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=500)

# Make a prediction
print(model.predict([5]))  # Should output a value close to 5
