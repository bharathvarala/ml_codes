import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Define your data and model
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = tf.Variable(1.0)

# Define the loss function
def loss(x, y):
    y_pred = x * w
    return tf.reduce_mean(tf.square(y_pred - y))

# Define the optimizer and learning rate
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Define the training loop
def train(x, y):
    with tf.GradientTape() as tape:
        l = loss(x, y)
    gradients = tape.gradient(l, [w])
    optimizer.apply_gradients(zip(gradients, [w]))

# Train the model for a certain number of epochs
for epoch in range(100):
    train(x_data, y_data)

# Plot the data points and the learned linear function
x_plot = np.linspace(0, 4, 10)
y_plot = w.numpy() * x_plot
plt.scatter(x_data, y_data)
plt.plot(x_plot, y_plot, color='red')
plt.show()
