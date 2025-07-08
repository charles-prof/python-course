# 🐵 Let's teach a monkey to tell apples from bananas! 🍎🍌

# We use some helpers to make numbers and draw pictures
import numpy as np
import matplotlib.pyplot as plt

# 1. 🍎🍌 Make pretend fruit data for our monkey
# Each fruit has 2 numbers (like size and color)
# Label: 0 = apple, 1 = banana
np.random.seed(42)
apples = np.random.randn(50, 2) + [2, 2]   # Apples are near (2,2)
bananas = np.random.randn(50, 2) + [6, 6]  # Bananas are near (6,6)
X = np.vstack([apples, bananas])           # All fruits together
y = np.array([0]*50 + [1]*50)              # 0 for apples, 1 for bananas

# 2. 🧠 Monkey's brain: a tiny neural network
# It has 2 input spots (for fruit features), 2 hidden spots, and 1 output spot
input_size = 2
hidden_size = 2
output_size = 1

# Randomly start the brain's connections (weights)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# 3. 🏃 Monkey learns by looking at fruits many times!
def sigmoid(x):
    # Squish numbers between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # How much does the squish change?
    s = sigmoid(x)
    return s * (1 - s)

# 4. 🏫 Training time! Show fruits to the monkey 1000 times
for step in range(1000):
    # Forward: Monkey looks at fruit and guesses
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Monkey's guess (0=apple, 1=banana)

    # Loss: How wrong was the monkey?
    loss = np.mean((y.reshape(-1, 1) - a2) ** 2)

    # Backward: Monkey learns from mistakes
    dz2 = a2 - y.reshape(-1, 1)
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(z1)
    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update: Monkey changes its brain a little bit
    lr = 0.1  # How fast monkey learns
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Print sometimes
    if step % 200 == 0:
        print(f"Step {step}, Monkey's mistake: {loss:.4f}")

# 5. 🖼️ Let's see how well the monkey learned!
def predict(X):
    # Monkey looks and guesses
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return (a2 > 0.5).astype(int)

# Draw apples and bananas, color shows monkey's guess
# Color by true label (apples=0, bananas=1), not by prediction
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("Monkey's Fruit Data (True Labels) 🍎🍌")
plt.xlabel("Fruit Feature 1")
plt.ylabel("Fruit Feature 2")
plt.show()
