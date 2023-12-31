import tensorflow as tf

print('''

*** Begin Program ***

''')

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

print('''

*** End Program ***

''')

'''
/home/lauren/Desktop/chester/virtualenv/env/lib/python3.11/site-packages/keras/src/layers/core/dense.py:73: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
Traceback (most recent call last):
  File "/home/lauren/Desktop/chester/ml/tensorflow_testing.py", line 22, in <module>
    model.fit(x_train, y_train, epochs=500)
  File "/home/lauren/Desktop/chester/virtualenv/env/lib/python3.11/site-packages/keras/src/utils/traceback_utils.py", line 123, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/lauren/Desktop/chester/virtualenv/env/lib/python3.11/site-packages/keras/src/trainers/data_adapters/__init__.py", line 113, in get_data_adapter
    raise ValueError(f"Unrecognized data type: x={x} (of type {type(x)})")
ValueError: Unrecognized data type: x=[1, 2, 3, 4] (of type <class 'list'>)
'''
